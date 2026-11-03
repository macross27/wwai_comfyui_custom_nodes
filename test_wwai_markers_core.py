"""Tests for the rules the marker nodes cannot be allowed to get wrong.

Runs anywhere: `python -m unittest` from the repo root, no ComfyUI, no torch.
"""

import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import types
import unittest
from unittest import mock

import wwai_markers_core
from wwai_markers_core import (
    MarkerContractError,
    check_staged_file,
    copy_file_verbatim,
    extension_of,
    require_marker_name,
    require_single,
    split_clicks,
    staged_file_delete,
    unannotated,
)


def synthetic_splat_ply():
    """A binary PLY carrying Gaussian Splat attributes a mesh writer would drop.

    The header names f_dc_*/opacity/scale_*/rot_* — the per-splat fields that
    survive a byte copy and do not survive a triangle-mesh round trip.
    """
    header = (
        b"ply\n"
        b"format binary_little_endian 1.0\n"
        b"element vertex 2\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        b"property float opacity\n"
        b"property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        b"property float rot_0\nproperty float rot_1\n"
        b"property float rot_2\nproperty float rot_3\n"
        b"end_header\n"
    )
    # 2 vertices x 14 float32 fields, plus a stray high byte so any text-mode
    # or re-encoding copy shows up as a difference.
    body = bytes(range(256)) * 2 + b"\x00\xff\xfe\x80"
    return header + body


class CopyFileVerbatimTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_splat_ply_survives_byte_for_byte_and_keeps_its_extension(self):
        src = os.path.join(self.tmp, "gaussian_splat.ply")
        payload = synthetic_splat_ply()
        with open(src, "wb") as handle:
            handle.write(payload)

        dest = os.path.join(self.tmp, "out", "WWAI_00001_.ply")
        returned = copy_file_verbatim(src, dest)

        self.assertEqual(returned, dest)
        self.assertTrue(dest.endswith(".ply"))
        with open(dest, "rb") as handle:
            self.assertEqual(handle.read(), payload)

    def test_copy_creates_the_destination_directory(self):
        src = os.path.join(self.tmp, "model.glb")
        with open(src, "wb") as handle:
            handle.write(b"glTF\x02\x00\x00\x00")

        dest = os.path.join(self.tmp, "deep", "nested", "model.glb")
        copy_file_verbatim(src, dest)

        self.assertTrue(os.path.isfile(dest))

    def test_copying_a_file_onto_itself_leaves_it_intact(self):
        src = os.path.join(self.tmp, "same.ply")
        payload = synthetic_splat_ply()
        with open(src, "wb") as handle:
            handle.write(payload)

        copy_file_verbatim(src, src)

        with open(src, "rb") as handle:
            self.assertEqual(handle.read(), payload)


class ExtensionOfTest(unittest.TestCase):
    def test_reads_the_real_extension_lowercased(self):
        self.assertEqual(extension_of("/tmp/Scene.PLY", "glb"), "ply")
        self.assertEqual(extension_of("/tmp/a.b/model.splat", "glb"), "splat")

    def test_falls_back_only_when_there_is_no_extension(self):
        self.assertEqual(extension_of("/tmp/noext", "glb"), "glb")


class RequireMarkerNameTest(unittest.TestCase):
    def test_returns_the_trimmed_name(self):
        self.assertEqual(require_marker_name("  prompt  ", "WWAIExposeText"), "prompt")

    def test_rejects_blank_and_whitespace_only_names(self):
        for blank in ("", "   ", None):
            with self.assertRaises(MarkerContractError):
                require_marker_name(blank, "WWAIExposeText")


class RequireSingleTest(unittest.TestCase):
    def test_returns_the_only_item(self):
        self.assertEqual(require_single(["a"], "WWAIExposeOutputImage", "image"), "a")

    def test_rejects_a_batch_and_says_how_many_it_got(self):
        with self.assertRaises(MarkerContractError) as caught:
            require_single(["a", "b"], "WWAIExposeOutputImage", "image")
        self.assertIn("got 2", str(caught.exception))

    def test_rejects_an_empty_result(self):
        with self.assertRaises(MarkerContractError):
            require_single([], "WWAIExposeOutputImage", "image")


HEX = hashlib.sha256(b"abc").hexdigest()
SECRET = "s" * 40


class CheckStagedFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        os.makedirs(os.path.join(self.tmp, "wwai"))

    def _put(self, rel, data):
        path = os.path.join(self.tmp, rel)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_right_bytes_pass(self):
        path = self._put(f"wwai/wwai-{HEX}.png", b"abc")
        check_staged_file(path, f"wwai/wwai-{HEX}.png")
        self.assertTrue(os.path.exists(path))

    def test_wrong_bytes_removed_and_raise(self):
        path = self._put(f"wwai/wwai-{HEX}.png", b"abd")
        with self.assertRaises(MarkerContractError) as ctx:
            check_staged_file(path, f"wwai/wwai-{HEX}.png")
        self.assertEqual(
            str(ctx.exception),
            f"WWAI_STAGED_FILE_UNUSABLE: wwai/wwai-{HEX}.png (mismatch)",
        )
        self.assertFalse(os.path.exists(path))

    def test_mismatch_still_typed_when_unlink_fails(self):
        path = self._put(f"wwai/wwai-{HEX}.png", b"abd")
        real = os.unlink
        os.unlink = lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked"))
        try:
            with self.assertRaises(MarkerContractError) as ctx:
                check_staged_file(path, f"wwai/wwai-{HEX}.png")
        finally:
            os.unlink = real
        self.assertEqual(
            str(ctx.exception),
            f"WWAI_STAGED_FILE_UNUSABLE: wwai/wwai-{HEX}.png (mismatch)",
        )

    def test_missing_raises(self):
        with self.assertRaises(MarkerContractError) as ctx:
            check_staged_file(
                os.path.join(self.tmp, f"wwai/wwai-{HEX}.png"), f"wwai/wwai-{HEX}.png"
            )
        self.assertIn("(missing)", str(ctx.exception))

    def test_root_author_file_unchecked(self):
        path = self._put("shot.png", b"anything")
        check_staged_file(path, "shot.png")
        check_staged_file(os.path.join(self.tmp, "gone.png"), "gone.png")
        self.assertTrue(os.path.exists(path))


class StagedFileDeleteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.input = os.path.join(self.tmp, "input")
        os.makedirs(os.path.join(self.input, "wwai"))
        self.secret_path = os.path.join(self.tmp, ".wwai_staged_file_secret")
        with open(self.secret_path, "w") as handle:
            handle.write(SECRET)

    def _call(self, filename, header=SECRET, queued=()):
        return staged_file_delete(
            self.input, self.secret_path, header, filename, lambda: set(queued)
        )

    def _put(self, name, data=b"x"):
        path = os.path.join(self.input, "wwai", name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def test_delete_then_absent(self):
        path = self._put(f"wwai-{HEX}.png")
        status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body), (200, {"wwai_staged_file": "deleted"}))
        self.assertFalse(os.path.exists(path))
        status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body), (404, {"wwai_staged_file": "absent"}))

    def test_renamed_copy_deleted(self):
        path = self._put(f"wwai-{HEX} (1).png")
        self.assertEqual(self._call(f"wwai-{HEX} (1).png")[0], 200)
        self.assertFalse(os.path.exists(path))

    def test_wrong_or_missing_header_is_403(self):
        path = self._put(f"wwai-{HEX}.png")
        for header in ("nope", "", None):
            status, body = self._call(f"wwai-{HEX}.png", header=header)
            self.assertEqual((status, body), (403, {"wwai_staged_file": "forbidden"}))
        self.assertTrue(os.path.exists(path))

    def test_secret_not_configured_is_503(self):
        for text in (None, "short", "  \n"):
            if text is None:
                os.remove(self.secret_path)
            else:
                with open(self.secret_path, "w") as handle:
                    handle.write(text)
            status, body = self._call(f"wwai-{HEX}.png")
            self.assertEqual(
                (status, body), (503, {"wwai_staged_file": "not_configured"})
            )

    def test_whitespace_padded_secret_accepted(self):
        with open(self.secret_path, "w") as handle:
            handle.write(f"\n  {SECRET}  \n")
        self._put(f"wwai-{HEX}.png")
        self.assertEqual(self._call(f"wwai-{HEX}.png")[0], 200)

    def test_bad_names_are_400_refused(self):
        bad = [
            f"wwai-{HEX} (0).png",
            f"wwai-{HEX}(1).png",
            f"wwai-{HEX} (01).png",
            "../x",
            f"wwai/wwai-{HEX}.png",
            f"wwai-{HEX.upper()}.png",
            f"wwai-{HEX[:63]}.png",
            f"wwai-{HEX}.PNG",
            f"wwai-{HEX}.png\n",
            f"wwai-{HEX}.png\x00",
            f"wwai-{HEX}\\.png",
            f"wwai-{HEX}",
            f"wwai-{HEX}." + "a" * 17,
            "",
            None,
            5,
        ]
        for name in bad:
            status, body = self._call(name)
            self.assertEqual(
                (status, body["wwai_staged_file"]), (400, "refused"), repr(name)
            )

    def test_non_regular_file_is_400(self):
        os.makedirs(os.path.join(self.input, "wwai", f"wwai-{HEX}.png"))
        status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body), (400, {"wwai_staged_file": "not_regular"}))

    def test_oserror_is_500_failed(self):
        self._put(f"wwai-{HEX}.png")
        real = os.unlink
        os.unlink = lambda *a, **k: (_ for _ in ()).throw(PermissionError("no"))
        try:
            status, body = self._call(f"wwai-{HEX}.png")
        finally:
            os.unlink = real
        self.assertEqual(
            (status, body), (500, {"wwai_staged_file": "failed", "detail": "no"})
        )

    def test_file_a_queued_prompt_names_is_409_in_use(self):
        path = self._put(f"wwai-{HEX}.png")
        status, body = self._call(
            f"wwai-{HEX}.png", queued={f"wwai/wwai-{HEX}.png", "other"}
        )
        self.assertEqual((status, body), (409, {"wwai_staged_file": "in_use"}))
        self.assertTrue(os.path.exists(path))
        # A different queued file, or the bare name without the folder, does not block.
        self.assertEqual(
            self._call(f"wwai-{HEX}.png", queued={f"wwai-{HEX}.png"})[0], 200
        )

    def test_queue_is_read_only_after_secret_and_name_checks(self):
        def boom():
            raise AssertionError("queue read before the checks")

        self.assertEqual(
            staged_file_delete(
                self.input, self.secret_path, "nope", f"wwai-{HEX}.png", boom
            )[0],
            403,
        )
        self.assertEqual(
            staged_file_delete(self.input, self.secret_path, SECRET, "../x", boom)[0],
            400,
        )

    def test_missing_folder_is_absent(self):
        os.rmdir(os.path.join(self.input, "wwai"))
        self.assertEqual(
            self._call(f"wwai-{HEX}.png"), (404, {"wwai_staged_file": "absent"})
        )

    def test_folder_that_is_a_file_is_400(self):
        os.rmdir(os.path.join(self.input, "wwai"))
        with open(os.path.join(self.input, "wwai"), "wb"):
            pass
        status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body["wwai_staged_file"]), (400, "not_regular"))
        self.assertIn("detail", body)

    @unittest.skipIf(sys.platform == "win32", "symlinks need privileges on Windows")
    def test_symlinked_folder_is_refused_and_target_kept(self):
        elsewhere = os.path.join(self.tmp, "elsewhere")
        os.makedirs(elsewhere)
        victim = os.path.join(elsewhere, f"wwai-{HEX}.png")
        with open(victim, "wb") as handle:
            handle.write(b"x")
        os.rmdir(os.path.join(self.input, "wwai"))
        os.symlink(elsewhere, os.path.join(self.input, "wwai"))
        status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body["wwai_staged_file"]), (400, "not_regular"))
        self.assertIn("detail", body)
        self.assertTrue(os.path.exists(victim))

    def test_utf8_bom_secret_accepted(self):
        with open(self.secret_path, "w", encoding="utf-8-sig") as handle:
            handle.write(SECRET + "\r\n")
        self._put(f"wwai-{HEX}.png")
        self.assertEqual(self._call(f"wwai-{HEX}.png")[0], 200)

    def test_utf16_secret_accepted(self):
        for encoding in ("utf-16", "utf-16-be"):
            with open(self.secret_path, "wb") as handle:
                bom = b"\xfe\xff" if encoding == "utf-16-be" else b""
                handle.write(bom + (SECRET + "\r\n").encode(encoding))
            self._put(f"wwai-{HEX}.png")
            self.assertEqual(self._call(f"wwai-{HEX}.png")[0], 200, encoding)

    def test_undecodable_secret_is_not_configured(self):
        with open(self.secret_path, "wb") as handle:
            handle.write(b"\xc3\x28" * 40)
        self.assertEqual(
            self._call(f"wwai-{HEX}.png"), (503, {"wwai_staged_file": "not_configured"})
        )

    def test_annotated_queued_value_is_in_use(self):
        self._put(f"wwai-{HEX}.png")
        for tag in (" [input]", " [output]", " [temp]"):
            status, body = self._call(
                f"wwai-{HEX}.png", queued={f"wwai/wwai-{HEX}.png{tag}"}
            )
            self.assertEqual((status, body), (409, {"wwai_staged_file": "in_use"}), tag)
        self.assertEqual(
            unannotated(f"wwai/wwai-{HEX}.png [input]"), f"wwai/wwai-{HEX}.png"
        )
        self.assertEqual(unannotated("plain.png"), "plain.png")

    def test_unreadable_queue_is_500_and_keeps_the_file(self):
        path = self._put(f"wwai-{HEX}.png")

        def broken():
            raise RuntimeError("queue gone")

        status, body = staged_file_delete(
            self.input, self.secret_path, SECRET, f"wwai-{HEX}.png", broken
        )
        self.assertEqual(
            (status, body),
            (
                500,
                {
                    "wwai_staged_file": "failed",
                    "detail": "cannot read the prompt queue: queue gone",
                },
            ),
        )
        self.assertTrue(os.path.exists(path))

    def test_surrogate_escaped_header_is_403(self):
        header = b"\xff\xfe".decode("utf-8", "surrogateescape") + SECRET
        self.assertEqual(
            self._call(f"wwai-{HEX}.png", header=header),
            (403, {"wwai_staged_file": "forbidden"}),
        )

    def _lstat_reporting_reparse(self, target, tag=0xA0000003):
        real = os.lstat

        def fake(path, **kwargs):
            st = real(path, **kwargs)
            if os.path.basename(path) != os.path.basename(target):
                return st
            return types.SimpleNamespace(
                st_mode=st.st_mode, st_file_attributes=0x400, st_reparse_tag=tag
            )

        return mock.patch.object(wwai_markers_core.os, "lstat", fake)

    def test_reparse_point_folder_is_refused(self):
        path = self._put(f"wwai-{HEX}.png")
        with self._lstat_reporting_reparse(os.path.join(self.input, "wwai")):
            status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body["wwai_staged_file"]), (400, "not_regular"))
        self.assertTrue(os.path.exists(path))

    def test_reparse_point_file_is_refused(self):
        path = self._put(f"wwai-{HEX}.png")
        self.assertTrue(stat.S_ISREG(os.lstat(path).st_mode))
        with self._lstat_reporting_reparse(path):
            status, body = self._call(f"wwai-{HEX}.png")
        self.assertEqual((status, body), (400, {"wwai_staged_file": "not_regular"}))
        self.assertTrue(os.path.exists(path))

    def test_cloud_placeholder_reparse_tag_is_not_refused(self):
        # OneDrive (0x9000001A) sets the reparse bit on a folder and a file: both are ordinary.
        folder_file = self._put(f"wwai-{HEX}.png")
        with self._lstat_reporting_reparse(
            os.path.join(self.input, "wwai"), tag=0x9000001A
        ):
            self.assertEqual(
                self._call(f"wwai-{HEX}.png"), (200, {"wwai_staged_file": "deleted"})
            )
        self.assertFalse(os.path.exists(folder_file))
        file_path = self._put(f"wwai-{'b' * 64}.png")
        with self._lstat_reporting_reparse(file_path, tag=0x9000001A):
            self.assertEqual(
                self._call(f"wwai-{'b' * 64}.png"),
                (200, {"wwai_staged_file": "deleted"}),
            )
        self.assertFalse(os.path.exists(file_path))


class SplitClicksTest(unittest.TestCase):
    """Every row of click_cases.json; WWAI runs the same rows through its own coercer."""

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "click_cases.json"),
        encoding="utf-8",
    ) as _cases_file:
        CASES = json.load(_cases_file)

    def test_valid_rows(self):
        for row in self.CASES["valid"]:
            with self.subTest(row["name"]):
                got = split_clicks(row["canonical"])
                names = ("include_points", "exclude_points", "include_boxes", "exclude_boxes", "frame")
                self.assertEqual(dict(zip(names, got)), row["outputs"])

    def test_valid_rows_accept_their_uncanonical_form(self):
        for row in self.CASES["valid"]:
            with self.subTest(row["name"]):
                self.assertEqual(split_clicks(json.dumps(row["state"])), split_clicks(row["canonical"]))

    def test_invalid_rows(self):
        for row in self.CASES["invalid"]:
            with self.subTest(row["name"]):
                with self.assertRaises(MarkerContractError) as caught:
                    split_clicks(row["state"])
                self.assertIn(row["refusal"], str(caught.exception))

    def test_non_text_is_refused(self):
        with self.assertRaises(MarkerContractError):
            split_clicks({"source": "a" * 64, "frame": 0, "items": []})


if __name__ == "__main__":
    unittest.main()
