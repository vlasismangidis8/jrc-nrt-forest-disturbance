"""Tests for scripts/download_data.py against a local HTTP server (no network needed)."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
)
dd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dd)

PAYLOAD = b"x" * 1000


class Handler(BaseHTTPRequestHandler):
    """Serves PAYLOAD for any path; the first `stall_first` requests hang instead."""

    stall_first = 0
    seen = 0
    lock = threading.Lock()

    def do_GET(self):
        with Handler.lock:
            Handler.seen += 1
            stall = Handler.seen <= Handler.stall_first
        try:
            if stall:
                time.sleep(2.0)  # longer than the patched read timeout
            if self.path.endswith("/missing"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        except OSError:
            pass  # the client gave up on a stalled request

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    Handler.seen = 0
    Handler.stall_first = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    monkeypatch.setattr(dd, "TIMEOUT", (2, 0.5))
    monkeypatch.setattr(dd, "STALL_COOLDOWN", 0.2)
    monkeypatch.setattr(dd, "MAX_STALLS", 2)
    monkeypatch.setattr(dd, "THROTTLE", dd.Throttle())
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def make_manifest(names):
    return {
        "psu": 1,
        "bands": None,
        "store": "data/sentinel2/PSU_0000001.zarr",
        "zmetadata": {
            "zarr_consolidated_format": 1,
            "metadata": {
                ".zgroup": {"zarr_format": 2},
                "r10m/B04/.zarray": {"shape": [len(names)], "chunks": [1]},
            },
        },
        "arrays": {
            "r10m/B04": {"expected_chunks": len(names), "files": dict.fromkeys(names, 1000.0)}
        },
        "complete": False,
    }


def test_fetch_downloads_then_skips(server, tmp_path):
    dest = tmp_path / "r10m" / "0.0.0"
    assert dd.fetch(f"{server}/f", dest) == ("downloaded", 1000)
    assert dest.read_bytes() == PAYLOAD
    assert dd.fetch(f"{server}/f", dest) == ("skipped", 1000)
    assert Handler.seen == 1
    assert not list(tmp_path.rglob("*.part"))


def test_fetch_survives_a_temporary_stall(server, tmp_path):
    Handler.stall_first = 1
    dest = tmp_path / "0.0.0"
    assert dd.fetch(f"{server}/f", dest) == ("downloaded", 1000)
    assert not dd.THROTTLE.aborted


def test_fetch_gives_up_when_the_server_stays_silent(server, tmp_path):
    Handler.stall_first = 10**6
    dest = tmp_path / "0.0.0"
    with pytest.raises(dd.ServerUnresponsive):
        dd.fetch(f"{server}/f", dest)
    assert dd.THROTTLE.aborted
    assert not dest.exists()
    assert not list(tmp_path.rglob("*.part"))


def test_http_error_is_a_failure_not_a_stall(server, tmp_path, monkeypatch):
    monkeypatch.setattr(dd.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="404"):
        dd.fetch(f"{server}/missing", tmp_path / "x", attempts=2)
    assert Handler.seen == 2
    assert not dd.THROTTLE.aborted


def test_cube_is_complete_when_every_file_is_on_disk(server, tmp_path):
    (tmp_path / "manifests").mkdir()
    manifest = make_manifest(["0", "1"])
    nbytes, failures = dd.download_cube(f"{server}/", tmp_path, manifest, workers=2)
    assert (nbytes, failures) == (2000, [])
    store = tmp_path / "sentinel2" / "PSU_0000001.zarr"
    assert (store / "r10m" / "B04" / "1").read_bytes() == PAYLOAD
    assert json.loads((store / ".zgroup").read_text()) == {"zarr_format": 2}
    assert json.loads((store / ".zmetadata").read_text()) == manifest["zmetadata"]
    assert json.loads((tmp_path / "manifests" / "PSU_0000001.json").read_text())["complete"]


def test_cube_is_not_complete_after_the_server_went_silent(server, tmp_path):
    Handler.stall_first = 10**6
    (tmp_path / "manifests").mkdir()
    manifest = make_manifest(["0", "1", "2"])
    dd.download_cube(f"{server}/", tmp_path, manifest, workers=2)
    assert dd.THROTTLE.aborted
    assert not manifest["complete"]
    assert not (tmp_path / "manifests" / "PSU_0000001.json").exists()


def test_index_rows_are_parsed_and_navigation_links_ignored():
    page = (
        '<tr><th><a href="?C=N;O=D">Name</a></th></tr>'
        '<tr><td valign="top"><img src="/icons/back.gif" alt="[PARENTDIR]"></td>'
        '<td><a href="/ftp/x/">Parent Directory</a></td><td>&nbsp;</td>'
        '<td align="right">  - </td><td>&nbsp;</td></tr>'
        '<tr><td valign="top"><img src="/icons/text.gif" alt="[TXT]"></td>'
        '<td><a href="copyright.txt">copyright.txt</a></td><td align="right">2026-03-05 12:26  </td>'
        '<td align="right">540 </td><td>&nbsp;</td></tr>'
        '<tr><td valign="top"><img src="/icons/unknown.gif" alt="[   ]"></td>'
        '<td><a href="0.0.10">0.0.10</a></td><td align="right">2026-06-03 18:45  </td>'
        '<td align="right"> 30K</td><td>&nbsp;</td></tr>'
        '<tr><td valign="top"><img src="/icons/folder.gif" alt="[DIR]"></td>'
        '<td><a href="r10m/">r10m/</a></td><td align="right">2026-06-06 03:08  </td>'
        '<td align="right">  - </td><td>&nbsp;</td></tr>'
    )
    assert dd.INDEX_ROW.findall(page) == [
        ("copyright.txt", "540", ""),
        ("0.0.10", "30", "K"),
        ("r10m/", "", ""),
    ]
    assert dd.CHUNK_NAME.match("0.0.10") and not dd.CHUNK_NAME.match("copyright.txt")


def test_band_subset_drops_metadata_of_arrays_that_are_not_mirrored():
    meta = {
        ".zgroup": {},
        ".zattrs": {},
        "r10m/.zgroup": {},
        "r10m/B04/.zarray": {},
        "r10m/B04/.zattrs": {},
        "r10m/B02/.zarray": {},
        "r10m/time/.zarray": {},
        "r20m/.zgroup": {},
        "r20m/B05/.zarray": {},
        "r20m/SCL/.zarray": {},
    }
    kept = dd.select_metadata(meta, ["r10m/B04", "r10m/time", "r20m/SCL"])
    assert set(kept) == {
        ".zgroup",
        ".zattrs",
        "r10m/.zgroup",
        "r10m/B04/.zarray",
        "r10m/B04/.zattrs",
        "r10m/time/.zarray",
        "r20m/.zgroup",
        "r20m/SCL/.zarray",
    }


def test_expected_chunk_count():
    assert dd.n_expected_chunks({"shape": [617, 542, 542], "chunks": [10, 50, 50]}) == 7502
    assert dd.n_expected_chunks({"shape": [], "chunks": []}) == 1
