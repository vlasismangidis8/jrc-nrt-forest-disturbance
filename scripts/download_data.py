#!/usr/bin/env python3
"""Resumable mirror of the JRC NRT challenge data into data/.

Mirrors, 1:1 with the JRC FTP layout:
  aux    PSU boundaries, demo PSU/SSU FlatGeobuf files, demo reference labels
  masks  Tree Cover masks (10/20/30 m) for the selected PSUs
  cubes  Sentinel-2 L2A Zarr stores (raw chunk files) for the selected PSUs

Examples (from the repo root):
  uv run python scripts/download_data.py --psus demo --dry-run   # sizes only
  uv run python scripts/download_data.py --psus demo
  uv run python scripts/download_data.py --psus 4914 28788 --bands B04 B08 SCL
  uv run python scripts/download_data.py --psus all --bands B04 B08 B8A B11 B12 SCL

Resumability: every file is written to "<name>.part" and renamed once its size
matches Content-Length, so an existing file is always complete and is skipped.
A cube is flagged complete in data/manifests/ only when every listed file is on
disk; a complete cube is skipped without any request. The Zarr metadata of a
cube is written at that moment, so an incomplete cube cannot be opened.

Server stalls: the JRC server sometimes accepts connections and then stays
silent for minutes. All workers then pause together and retry; after
--max-stalls silent cool-downs in a row the run stops with exit code 2.
For a gentle unattended run:
  --workers 1 --max-rate 5 --stall-cooldown 300 --max-stalls 100

Exit codes: 0 all done, 1 some files failed (see data/download_failures.txt),
2 stopped early because the server is unresponsive. Re-run to resume.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import truststore
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Verify TLS against the operating-system certificate store. Required on machines
# where an antivirus re-signs HTTPS traffic; equivalent to certifi elsewhere.
truststore.inject_into_ssl()

REPO = Path(__file__).resolve().parents[1]
TIMEOUT = (20, 90)  # connect, read (the server can take ~45 s to send the first byte)
STALL_COOLDOWN = 120  # seconds every worker pauses after a timeout or dropped connection
MAX_STALLS = 10  # consecutive cool-downs without one successful download before giving up
UNITS = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
# One row of an Apache "Index of" table: href, then size ("540", "26K", "1.4M") or "-" for folders.
INDEX_ROW = re.compile(
    r'<a href="([^"?/][^"]*)">[^<]*</a></td><td[^>]*>[^<]*</td><td[^>]*>\s*(?:([0-9.]+)([KMG]?)|-)\s*</td>'
)
CHUNK_NAME = re.compile(r"^\d+(\.\d+)*$")
COORD_ARRAYS = {"time", "x", "y", "spatial_ref"}

_local = threading.local()
_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ServerUnresponsive(RuntimeError):
    """The server stayed silent through MAX_STALLS cool-downs; the run stops early."""


class Throttle:
    """Shared back-off for all worker threads.

    The JRC server sometimes accepts connections but sends nothing for minutes.
    When one worker sees a timeout, every worker pauses for STALL_COOLDOWN
    instead of piling more requests onto a server that is not answering.
    `max_rate` caps the number of requests per second across all workers.
    """

    def __init__(self, max_rate: float | None = None) -> None:
        self._lock = threading.Lock()
        self._resume_at = 0.0
        self._next_slot = 0.0
        self._stalls = 0
        self.max_rate = max_rate
        self.aborted = False

    def wait(self) -> None:
        """Block until this thread may send one request."""
        while True:
            if self.aborted:
                raise ServerUnresponsive("server unresponsive, run stopped early")
            delay = self._resume_at - time.time()
            if delay <= 0:
                break
            time.sleep(min(delay, 5.0))
        if self.max_rate:
            with self._lock:
                now = time.time()
                slot = max(self._next_slot, now)
                self._next_slot = slot + 1.0 / self.max_rate
            if slot > now:
                time.sleep(slot - now)

    def success(self) -> None:
        self._stalls = 0

    def stalled(self, error: Exception) -> None:
        with self._lock:
            now = time.time()
            if now < self._resume_at or self.aborted:
                return  # another worker already started this cool-down
            self._stalls += 1
            if self._stalls > MAX_STALLS:
                self.aborted = True
                log(
                    f"server silent through {MAX_STALLS} cool-downs: stopping. Re-run later to resume."
                )
                return
            self._resume_at = now + STALL_COOLDOWN
            log(
                f"server not responding ({brief(error)}): all workers pause "
                f"{STALL_COOLDOWN:g} s (stall {self._stalls}/{MAX_STALLS})"
            )


THROTTLE = Throttle()


def brief(error: Exception) -> str:
    """The informative tail of a requests/urllib3 error message, for one log line."""
    text = str(error)
    cut = text.find("Caused by")
    return f"{type(error).__name__}: {text[cut:] if cut >= 0 else text}"[:140]


def is_timeout(error: Exception) -> bool:
    """True when the server stayed silent; False when the connection was dropped.

    The two need opposite reactions. Silence means the server is stuck, so all
    workers back off. A dropped keep-alive connection is routine (Apache closes
    idle ones after 5 s) and only needs a fresh connection.
    """
    return isinstance(error, requests.Timeout) or "timed out" in str(error).lower()


def on_connection_error(error: Exception, drops: int) -> int:
    """React to a failed request; returns the updated count of consecutive drops."""
    if is_timeout(error) or drops >= 3:
        THROTTLE.stalled(error)
        return 0
    s = getattr(_local, "session", None)
    if s is not None:
        s.close()  # the pooled connection is dead; the next request opens a new one
    time.sleep(1.0 + drops)
    return drops + 1


def session() -> requests.Session:
    """One keep-alive session per thread (connection reuse matters on this server)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        # read=0: read errors surface at once and are sorted by on_connection_error().
        retry = Retry(
            total=3,
            read=0,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1))
        s.headers["User-Agent"] = "nrt-challenge-mirror/0.1 (python-requests)"
        _local.session = s
    return s


def http_get(url: str) -> requests.Response:
    """GET a small resource through the shared throttle, waiting out server stalls."""
    drops = 0
    while True:
        THROTTLE.wait()
        try:
            r = session().get(url, timeout=TIMEOUT)
        except (requests.Timeout, requests.ConnectionError) as e:
            drops = on_connection_error(e, drops)
            continue
        r.raise_for_status()
        THROTTLE.success()
        return r


def list_dir(url: str) -> list[tuple[str, bool, float]]:
    """Parse an Apache directory index into (name, is_dir, approximate_size_bytes)."""
    r = http_get(url)
    entries = []
    for href, num, unit in INDEX_ROW.findall(r.text):
        name = html.unescape(href)
        size = float(num) * UNITS[unit] if num else 0.0
        entries.append((name.rstrip("/"), name.endswith("/"), size))
    return entries


def fetch(url: str, dest: Path, attempts: int = 4) -> tuple[str, int]:
    """Download url to dest unless dest already exists. Returns (status, bytes)."""
    if dest.exists():
        return "skipped", dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    errors = 0  # stalls are not counted here: Throttle bounds them globally
    drops = 0
    while errors < attempts:
        THROTTLE.wait()
        try:
            with session().get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                n = 0
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1 << 16):
                        f.write(block)
                        n += len(block)
                expected = r.headers.get("Content-Length")
                if (
                    expected is not None
                    and "Content-Encoding" not in r.headers
                    and int(expected) != n
                ):
                    raise OSError(f"size mismatch: got {n} bytes, expected {expected}")
            os.replace(tmp, dest)
            THROTTLE.success()
            return "downloaded", n
        except (requests.Timeout, requests.ConnectionError) as e:
            tmp.unlink(missing_ok=True)
            drops = on_connection_error(e, drops)
        except (requests.RequestException, OSError) as e:
            last_error = e
            errors += 1
            tmp.unlink(missing_ok=True)
            time.sleep(2.0 * errors)
    raise RuntimeError(f"{url}: {last_error}")


def run_tasks(
    tasks: list[tuple[str, Path]], workers: int, label: str
) -> tuple[int, int, list[str]]:
    """Fetch (url, dest) pairs concurrently. Returns (bytes_on_disk, n_downloaded, failures)."""
    total_bytes = n_new = done = 0
    failures: list[str] = []
    t0 = last = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, url, dest): url for url, dest in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                status, n = fut.result()
                total_bytes += n
                n_new += status == "downloaded"
            except ServerUnresponsive:
                pass  # reported once by main(); these files are simply not fetched yet
            except Exception as e:  # noqa: BLE001 - recorded and reported, run continues
                failures.append(str(e))
            now = time.time()
            if now - last >= 30:
                rate = done / (now - t0)
                eta = (len(tasks) - done) / rate if rate > 0 else float("nan")
                log(
                    f"{label}: {done}/{len(tasks)} files, {total_bytes / 1e6:.0f} MB, "
                    f"{rate:.0f} files/s, ETA {eta / 60:.1f} min, {len(failures)} failed"
                )
                last = now
    return total_bytes, n_new, failures


def n_expected_chunks(zarray: dict) -> int:
    return math.prod(math.ceil(s / c) for s, c in zip(zarray["shape"], zarray["chunks"]))


def select_metadata(meta: dict, arrays: list[str]) -> dict:
    """Consolidated-metadata entries of the root, the groups and the given arrays only.

    Keys look like ".zgroup" (root), "r10m/.zattrs" (group) or "r10m/B04/.zarray"
    (array). Dropping the entries of arrays that are not mirrored means a band
    subset can never be opened as arrays that silently read back as fill values.
    """
    groups = {a.rsplit("/", 1)[0] for a in arrays if "/" in a}

    def keep(key: str) -> bool:
        parent = key.rsplit("/", 1)[0] if "/" in key else ""
        return parent == "" or parent in arrays or parent in groups

    return {k: v for k, v in meta.items() if keep(k)}


def finish_manifest(manifest: dict) -> dict:
    manifest["approx_bytes"] = sum(sum(v["files"].values()) for v in manifest["arrays"].values())
    manifest["n_files"] = sum(len(v["files"]) for v in manifest["arrays"].values())
    return manifest


def subset_manifest(manifest: dict, bands: list[str]) -> dict:
    """Restrict a cached listing to `bands` (plus coordinates) without asking the server."""
    names = set(bands) | COORD_ARRAYS
    arrays = {a: v for a, v in manifest["arrays"].items() if a.split("/")[-1] in names}
    zmeta = manifest["zmetadata"]
    return finish_manifest(
        {
            "psu": manifest["psu"],
            "bands": bands,
            "store": manifest["store"],
            "zmetadata": {**zmeta, "metadata": select_metadata(zmeta["metadata"], list(arrays))},
            "arrays": arrays,
            "complete": False,
        }
    )


def plan_cube(
    base: str, cfg: dict, data_dir: Path, psu: int, bands: list[str] | None, workers: int
) -> dict:
    """List every chunk file of one PSU cube. The plan is cached in data/manifests/."""
    manifest_path = data_dir / "manifests" / f"PSU_{psu:07d}.json"
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        have = cached.get("bands")
        covers = have is None or (bands is not None and set(bands) <= set(have))
        if have == bands or (covers and cached.get("complete")):
            return cached  # same selection, or a finished mirror that already contains it
        if covers:
            return subset_manifest(cached, bands)

    store_rel = cfg["remote"]["s2_cube"].format(psu=psu)
    store_url = f"{base}{store_rel}/"
    zmeta = http_get(store_url + ".zmetadata").json()
    meta = zmeta["metadata"]
    arrays = sorted(k[: -len("/.zarray")] for k in meta if k.endswith("/.zarray"))
    if bands is not None:
        unknown = set(bands) - {a.split("/")[-1] for a in arrays}
        if unknown:
            raise SystemExit(f"PSU {psu}: bands not in cube: {sorted(unknown)}")
        arrays = [a for a in arrays if a.split("/")[-1] in set(bands) | COORD_ARRAYS]

    def list_array(array: str) -> tuple[str, dict[str, float]]:
        files = {
            n: s
            for n, is_dir, s in list_dir(f"{store_url}{array}/")
            if not is_dir and CHUNK_NAME.match(n)
        }
        return array, files

    with ThreadPoolExecutor(max_workers=workers) as pool:
        listed = dict(pool.map(list_array, arrays))

    manifest = finish_manifest(
        {
            "psu": psu,
            "bands": bands,
            "store": store_rel,
            "zmetadata": {**zmeta, "metadata": select_metadata(meta, arrays)},
            "arrays": {
                a: {"expected_chunks": n_expected_chunks(meta[f"{a}/.zarray"]), "files": listed[a]}
                for a in arrays
            },
            "complete": False,
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_cube_metadata(store_dir: Path, manifest: dict) -> None:
    """Write the hidden Zarr v2 metadata files (.zgroup/.zattrs/.zarray/.zmetadata).

    Apache hides dot-files from its listings; the consolidated .zmetadata holds
    an exact copy of every one of them, so they are written from it.
    """
    zmeta = manifest["zmetadata"]
    for key, content in zmeta["metadata"].items():
        path = store_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=4), encoding="utf-8")
    (store_dir / ".zmetadata").write_text(json.dumps(zmeta, indent=4), encoding="utf-8")


def download_cube(base: str, data_dir: Path, manifest: dict, workers: int) -> tuple[int, list[str]]:
    psu = manifest["psu"]
    store_dir = data_dir / Path(manifest["store"]).relative_to("data")
    store_url = f"{base}{manifest['store']}/"
    tasks = [
        (f"{store_url}{array}/{name}", store_dir / array / name)
        for array, info in manifest["arrays"].items()
        for name in info["files"]
    ]
    nbytes, n_new, failures = run_tasks(tasks, workers, f"PSU {psu:07d}")
    # "Complete" is decided by what is on disk, never by the absence of error reports.
    n_missing = sum(not dest.exists() for _, dest in tasks)
    if n_missing == 0:
        # The Zarr metadata is written last: a store with missing chunks stays
        # unopenable instead of reading back fill values where data should be.
        write_cube_metadata(store_dir, manifest)
        manifest["complete"] = True
        manifest["bytes"] = nbytes
        path = data_dir / "manifests" / f"PSU_{psu:07d}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    log(
        f"PSU {psu:07d}: {len(tasks)} files, {nbytes / 1e9:.2f} GB on disk, {n_new} new, "
        f"{n_missing} still missing, {len(failures)} failed"
    )
    return nbytes, failures


def resolve_psus(arg: list[str], cfg: dict, base: str) -> list[int]:
    if arg == ["demo"]:
        return list(cfg["demo_psus"])
    if arg == ["all"]:
        names = [n for n, is_dir, _ in list_dir(base + cfg["remote"]["s2_listing"]) if is_dir]
        return sorted(int(m.group(1)) for n in names if (m := re.fullmatch(r"PSU_(\d+)\.zarr", n)))
    return [int(a) for a in arg]


def main() -> int:
    global STALL_COOLDOWN, MAX_STALLS
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--psus", nargs="+", default=["demo"], help='"demo", "all", or PSU ids')
    ap.add_argument(
        "--what", nargs="+", default=["aux", "masks", "cubes"], choices=["aux", "masks", "cubes"]
    )
    ap.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="subset of cube arrays, e.g. B04 B08 SCL (default: all)",
    )
    ap.add_argument("--workers", type=int, default=4, help="parallel connections (default 4)")
    ap.add_argument(
        "--max-rate", type=float, default=None, help="cap on requests per second, all workers"
    )
    ap.add_argument(
        "--stall-cooldown",
        type=float,
        default=STALL_COOLDOWN,
        help=f"seconds to pause when the server stops answering (default {STALL_COOLDOWN})",
    )
    ap.add_argument(
        "--max-stalls",
        type=int,
        default=MAX_STALLS,
        help=f"silent cool-downs in a row before stopping (default {MAX_STALLS})",
    )
    ap.add_argument("--data-dir", type=Path, default=REPO / "data")
    ap.add_argument("--config", type=Path, default=REPO / "config" / "challenge.yaml")
    ap.add_argument(
        "--dry-run", action="store_true", help="list cubes and report approximate sizes only"
    )
    args = ap.parse_args()

    STALL_COOLDOWN, MAX_STALLS = args.stall_cooldown, args.max_stalls
    THROTTLE.max_rate = args.max_rate

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = cfg["base_url"]
    remote = cfg["remote"]
    data_dir: Path = args.data_dir
    bands = sorted(args.bands) if args.bands else None
    psus = resolve_psus(args.psus, cfg, base)
    log(f"{len(psus)} PSU(s): {psus if len(psus) <= 10 else str(psus[:10]) + ' ...'}")
    failures: list[str] = []

    def local(rel: str) -> Path:
        # remote "data/x" -> data/x ; remote "demo/x" -> data/demo/x
        return data_dir / (Path(rel).relative_to("data") if rel.startswith("data/") else Path(rel))

    small: list[tuple[str, Path]] = []
    if "aux" in args.what:
        for key in ("psu_boundaries", "demo_psus", "demo_ssus", "copyright"):
            small.append((base + remote[key], local(remote[key])))
        for idx in range(cfg["demo_site_count"]):
            rel = remote["demo_labels"].format(idx=idx)
            small.append((base + rel, local(rel)))
    if "masks" in args.what:
        for res in cfg["mask_resolutions"]:
            for psu in psus:
                rel = remote["forest_mask"].format(res=res, psu=psu)
                small.append((base + rel, local(rel)))
    if small and not args.dry_run:
        nbytes, n_new, fails = run_tasks(small, args.workers, "aux/masks")
        failures += fails
        log(
            f"aux/masks: {len(small)} files, {nbytes / 1e6:.1f} MB on disk, {n_new} new, {len(fails)} failed"
        )

    if "cubes" in args.what:
        total = 0.0
        for psu in psus:
            if THROTTLE.aborted:
                break
            try:
                manifest = plan_cube(base, cfg, data_dir, psu, bands, args.workers)
            except Exception as e:  # noqa: BLE001 - keep going with the other PSUs
                failures.append(f"PSU {psu}: planning failed: {e}")
                log(failures[-1])
                continue
            absent = {
                a: v["expected_chunks"] - len(v["files"])
                for a, v in manifest["arrays"].items()
                if v["expected_chunks"] != len(v["files"])
            }
            log(
                f"PSU {psu:07d}: {manifest['n_files']} files, ~{manifest['approx_bytes'] / 1e9:.2f} GB"
                + (f", chunk-count differences vs metadata: {absent}" if absent else "")
                + (" [already complete]" if manifest.get("complete") else "")
            )
            if args.dry_run:
                total += manifest["approx_bytes"]
            elif manifest.get("complete"):
                total += manifest.get("bytes", 0)
            else:
                nbytes, fails = download_cube(base, data_dir, manifest, args.workers)
                total += nbytes
                failures += fails
        log(
            f"cubes total: {total / 1e9:.2f} GB ({'approximate, from listings' if args.dry_run else 'on disk'})"
        )

    if failures:
        fail_path = data_dir / "download_failures.txt"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fail_path, "a", encoding="utf-8") as f:
            f.write(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "\n".join(failures) + "\n")
        log(f"{len(failures)} failure(s), see {fail_path}. Re-run the same command to retry.")
    if THROTTLE.aborted:
        log("stopped early: the server is not responding. Re-run the same command to resume.")
        return 2
    if failures:
        return 1
    log("done, no failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
