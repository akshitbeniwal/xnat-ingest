"""No server digest means size verification, never filename-only completion."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fileformats.generic import File

from xnat_ingest.api import check_upload_api, upload_api
from xnat_ingest.exceptions import IncompleteCheckumsException
from xnat_ingest.helpers.remotes import (
    FileChecksums,
    LocalSessionListing,
    S3SessionListing,
    SessionOnlyListing,
    compare_resource_with_xnat,
    file_sizes,
    get_xnat_checksums,
    get_xnat_resource,
)
from xnat_ingest.helpers.tests.test_all_uploaded_completeness import FakeXnat
from xnat_ingest.model.session import ImagingSession


def inventory(size=4, digest="", name="nested/a.bin"):
    return FileChecksums({name: digest}, sizes={} if size is None else {name: size})


@pytest.mark.parametrize("size", [0, 4])
def test_matching_sizes_allow_undigested_files(size):
    comparison = compare_resource_with_xnat(
        {"nested/a.bin": "md5"}, inventory(size), local_sizes={"nested/a.bin": size}
    )
    assert comparison.complete
    assert comparison.without_checksums == {"nested/a.bin"}


def test_same_name_truncated_file_is_not_complete():
    comparison = compare_resource_with_xnat(
        {"nested/a.bin": "md5"}, inventory(1), local_sizes={"nested/a.bin": 400}
    )
    assert not comparison.complete and not comparison.repairable
    assert comparison.size_differing == comparison.differing == {"nested/a.bin"}


@pytest.mark.parametrize("side", ["local", "remote"])
@pytest.mark.parametrize("size", [None, -1, True, 4.0])
def test_unknown_or_invalid_sizes_do_not_verify_a_file(side, size):
    local = (
        {}
        if side == "local" and size is None
        else {"nested/a.bin": size if side == "local" else 4}
    )
    remote = inventory(size if side == "remote" else 4)
    comparison = compare_resource_with_xnat(
        {"nested/a.bin": "md5"}, remote, local_sizes=local
    )
    assert not comparison.complete
    assert comparison.unverified == {"nested/a.bin"}


def test_partial_digests_verify_each_file_independently():
    remote = FileChecksums({"old": "ok", "new": ""}, sizes={"new": 1})
    comparison = compare_resource_with_xnat(
        {"old": "ok", "new": "md5"}, remote, local_sizes={"new": 4}
    )
    assert comparison.comparable and not comparison.complete
    assert comparison.differing == {"new"}


@pytest.mark.parametrize("size", ["4", 4, "0"])
def test_listing_retains_sizes_from_the_same_request(size):
    connection = MagicMock()
    connection.get.return_value.status_code = 200
    connection.get.return_value.json.return_value = {
        "ResultSet": {
            "Result": [
                {
                    "Name": "a.bin",
                    "URI": "/data/experiments/E/resources/7/files/nested/a.bin",
                    "digest": "",
                    "Size": size,
                }
            ]
        }
    }
    result = get_xnat_checksums(
        SimpleNamespace(xnat_session=connection, uri="/resource", id="7")
    )
    assert result == {"nested/a.bin": ""}
    assert result.sizes == {"nested/a.bin": int(size)}
    connection.get.assert_called_once()


@pytest.mark.parametrize("size", [-1, "-1", "unknown", True, 1.5])
def test_invalid_remote_sizes_fail(size):
    connection = MagicMock()
    connection.get.return_value.status_code = 200
    connection.get.return_value.json.return_value = {
        "ResultSet": {
            "Result": [
                {
                    "Name": "a",
                    "URI": "/data/experiments/E/resources/7/files/a",
                    "digest": "",
                    "Size": size,
                }
            ]
        }
    }
    with pytest.raises(ValueError):
        get_xnat_checksums(
            SimpleNamespace(xnat_session=connection, uri="/resource", id="7")
        )


class SizedXnat(FakeXnat):
    def __init__(self, *args, sizes, **kwargs):
        super().__init__(*args, **kwargs)
        self.sizes = sizes

    def get(self, uri):
        result = super().get(uri)
        for row in result.json()["ResultSet"]["Result"]:
            row.pop("Size", None)
            if row["Name"] in self.sizes:
                row["Size"] = self.sizes[row["Name"]]
        return result


@pytest.mark.parametrize("mode", ["local", "session-only", "s3", "relabelled", "split"])
@pytest.mark.parametrize("remote_size", [4, 1, None])
def test_every_fast_skip_requires_size_evidence(tmp_path, mode, remote_size):
    path = "DATA" if mode == "session-only" else "1.scan_one/DATA"
    names = ["nested/a.bin", "other/a.bin"]
    checksums = {name: hashlib.md5(b"data").hexdigest() for name in names}
    manifest = {"checksums": checksums, "datatype": "generic/file-set"}
    if mode == "s3":
        bucket = MagicMock()

        def download(key, stream):
            assert key == "manifest", "fast skip must not download file data"
            stream.write(json.dumps(manifest).encode())

        bucket.download_fileobj.side_effect = download
        objects = [
            (path.split("/") + ["__MANIFEST__.json"], SimpleNamespace(key="manifest"))
        ]
        objects += [
            (path.split("/") + name.split("/"), SimpleNamespace(size=4))
            for name in names
        ]
        listing = S3SessionListing(
            name="proj.subj.sess",
            objects=objects,
            bucket=bucket,
            cache_path=tmp_path / "cache",
        )
    else:
        session_dir = tmp_path / (
            "SESSLABEL" if mode == "session-only" else "proj.subj.sess"
        )
        resource_dir = session_dir / path
        for name in names:
            file = resource_dir / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"data")
        (resource_dir / "__MANIFEST__.json").write_text(json.dumps(manifest))
        listing = (
            SessionOnlyListing(session_dir)
            if mode == "session-only"
            else LocalSessionListing(session_dir)
        )
    held = {name: "" for name in names}
    kwargs = {}
    if mode == "session-only":
        scans = []
        kwargs = {
            "session_resources": {"DATA": held},
            "experiments": [
                {"ID": "E1", "label": "SESSLABEL", "xsiType": "xnat:mrSessionData"}
            ],
        }
    elif mode == "split":
        scans = [
            ("1-MR", "scan_one", {"DATA": {names[0]: ""}}),
            ("1-PR", "scan_one", {"secondary": {names[1]: ""}}),
        ]
    else:
        scans = [
            ("1", "scan_one", {"secondary" if mode == "relabelled" else "DATA": held})
        ]
    connection = SizedXnat(
        scans,
        sizes={} if remote_size is None else {name: remote_size for name in names},
        **kwargs,
    )
    assert listing.all_uploaded(connection) is (remote_size == 4)
    if mode == "s3":
        assert not listing._downloaded
        assert bucket.download_fileobj.call_count == 1


@pytest.mark.parametrize("scan_level", [False, True])
@pytest.mark.parametrize("remote_size", [1, None])
def test_bad_sizes_raise_instead_of_reporting_already_uploaded(
    tmp_path, scan_level, remote_size
):
    local = tmp_path / "a.bin"
    local.write_bytes(b"data")
    remote = object()
    session = SimpleNamespace(
        label="test",
        resources={"DATA": remote},
        xnat_session=SimpleNamespace(classes=SimpleNamespace()),
        scans={"1": SimpleNamespace(resources={"DATA": remote}, files=["a"])},
    )
    resource = SimpleNamespace(
        name="DATA",
        checksums={"a.bin": "md5"},
        fileset=File(local),
        metadata={},
        path="test:DATA",
        scan=SimpleNamespace(id="1", path="test:1") if scan_level else None,
    )
    with patch(
        "xnat_ingest.helpers.remotes.get_xnat_checksums",
        return_value=inventory(remote_size, name="a.bin"),
    ):
        with pytest.raises(IncompleteCheckumsException):
            get_xnat_resource(resource, session)


@pytest.mark.parametrize("check_checksums", [False, True])
@pytest.mark.parametrize("mode", ["healthy", "truncated", "unknown-size"])
def test_post_upload_verifies_sizes_even_when_hashing_disabled(
    tmp_path, caplog, check_checksums, mode
):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"complete original bytes")
    staged = ImagingSession(
        uid="test", project_id="proj", subject_id="subj", session_id="sess"
    )
    staged.add_resource("1", "raw", "DATA", File(payload))
    staging = tmp_path / "staging"
    (staging / "proj.subj.sess").mkdir(parents=True)
    held = {}
    remote = SimpleNamespace(
        upload=lambda path, name: held.update(
            {name: b"X" if mode == "truncated" else Path(path).read_bytes()}
        )
    )
    connection = MagicMock()
    connection.projects = {"proj": SimpleNamespace(experiments={})}
    connection.get_json.side_effect = lambda uri, query=None: {
        "ResultSet": {
            "Result": [{"ID": "proj", "name": "proj"}]
            if uri == "/data/archive/projects"
            else []
        }
    }

    def snapshot(_):
        return FileChecksums(
            {name: "" for name in held},
            sizes={}
            if mode == "unknown-size"
            else {name: len(data) for name, data in held.items()},
        )

    with (
        patch.object(upload_api.ImagingSession, "load", return_value=staged),
        patch.object(upload_api.FrameSet, "load", side_effect=KeyError("none")),
        patch.object(
            upload_api, "get_xnat_session", return_value=SimpleNamespace(id="E1")
        ),
        patch.object(upload_api, "get_xnat_resource", return_value=(remote, None)),
        patch.object(upload_api, "get_xnat_checksums", side_effect=snapshot),
    ):
        errors = upload_api.upload(
            str(staging),
            SimpleNamespace(connection=connection, server="fake://xnat"),
            always_include=["all"],
            check_checksums=check_checksums,
        )
    assert bool(errors) is (mode != "healthy")
    if mode != "healthy":
        assert "Successfully uploaded all files" not in caplog.text


def test_size_lookup_retains_nested_paths_and_follows_file_links(tmp_path):
    target = tmp_path / "actual"
    target.write_bytes(b"data")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/alias").symlink_to(target)
    assert file_sizes(tmp_path, ["nested/alias"]) == {"nested/alias": 4}
    for name in ["../outside", "/etc/passwd"]:
        with pytest.raises(ValueError):
            file_sizes(tmp_path, [name])


@pytest.mark.parametrize("remote_size", [4, 1, None])
def test_check_upload_reports_unverified_or_different_sizes(
    tmp_path, caplog, remote_size
):
    (tmp_path / "proj.subj.sess").mkdir()
    remote = object()
    scan = SimpleNamespace(files=["a.bin"], resources={"DATA": remote})
    session = SimpleNamespace(scans={"1": scan}, id="E1")
    connection = MagicMock()
    connection.projects = {
        "proj": SimpleNamespace(id="proj", experiments={"sess": session})
    }
    listing = SimpleNamespace(
        project_id="proj",
        session_id="sess",
        name="proj.subj.sess",
        resource_manifests={
            "1.raw/DATA": {"datatype": "generic/file", "checksums": {"a.bin": "md5"}}
        },
        resource_file_sizes=lambda resource, names: {name: 4 for name in names},
    )
    with (
        patch.object(
            check_upload_api,
            "Xnat",
            return_value=SimpleNamespace(connection=connection),
        ),
        patch.object(check_upload_api, "LocalSessionListing", return_value=listing),
        patch.object(
            check_upload_api,
            "get_xnat_checksums",
            return_value=inventory(remote_size, name="a.bin"),
        ),
    ):
        check_upload_api.check_upload(
            str(tmp_path), "fake://xnat", "test", "test", None, always_include=["all"]
        )
    assert ("CHECKSUM FAIL" in caplog.text) is (remote_size != 4)
