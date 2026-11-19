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
from fractions import Fraction
from unittest import mock

import wwai_markers_core
from wwai_markers_core import (
    MarkerContractError,
    check_staged_file,
    copy_file_verbatim,
    extension_of,
    frame_pattern,
    require_marker_name,
    require_single,
    require_storage_path,
    sequence_frames,
    write_in_place,
    check_keyframes,
    parse_keyframes,
    parse_ripple_manifest,
    ripple_grid_count,
    ripple_grid_frame,
    ripple_grid_slots,
    plan_tracking,
    staged_file_delete,
    tracked_frames,
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
    def test_a_clip_named_only_inside_a_queued_manifest_is_in_use(self):
        status, body = self._call(f"wwai-{'a' * 64}.mp4", queued={_ripple()})
        self.assertEqual((status, body["wwai_staged_file"]), (409, "in_use"))
        status, body = self._call(f"wwai-{'b' * 64}.png", queued={_ripple()})
        self.assertEqual((status, body["wwai_staged_file"]), (409, "in_use"))

    def test_a_manifest_naming_another_file_does_not_block(self):
        self._put(f"wwai-{'d' * 64}.mp4")
        self.assertEqual(self._call(f"wwai-{'d' * 64}.mp4", queued={_ripple()})[0], 200)

    def test_a_queued_text_that_is_no_manifest_is_ignored(self):
        self._put(f"wwai-{'a' * 64}.mp4")
        self.assertEqual(self._call(f"wwai-{'a' * 64}.mp4", queued={"{not json"})[0], 200)


class KeyframesTest(unittest.TestCase):
    """The keyframes grammar and the step count, from click_cases.json's keyframe keys."""

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "click_cases.json"),
        encoding="utf-8",
    ) as _cases_file:
        CASES = json.load(_cases_file)

    @staticmethod
    def _check(row):
        return check_keyframes(parse_keyframes(row["state"]), row["frame_count"], row["prompt_set"])

    def test_valid_rows(self):
        for row in self.CASES["keyframes_valid"]:
            with self.subTest(row["name"]):
                self._check(row)

    def test_invalid_rows_are_refused_by_name(self):
        reasons = set()
        for row in self.CASES["keyframes_invalid"]:
            with self.subTest(row["name"]):
                with self.assertRaises(MarkerContractError) as caught:
                    self._check(row)
                self.assertTrue(
                    str(caught.exception).startswith(f"{row['refusal']}: {row['reason']}:"),
                    str(caught.exception),
                )
                reasons.add(row["reason"])
        self.assertEqual(
            reasons,
            {
                "malformed", "in_after_out", "repeated_keyframe", "frame_outside_clip",
                "range_outside_clip", "picture_frame_not_zero", "prompt_keyframe_without_prompt",
                "nothing_to_track", "click_object_without_clicks", "empty_clip", "exclude_box_unsupported",
            },
        )

    def test_the_rows_the_plan_names_are_present(self):
        tracked = {row["name"]: row["tracked"] for row in self.CASES["tracked_frames"]}
        self.assertEqual(tracked["staggered keys over 40-120"], 204)
        self.assertEqual(tracked["prompt with no object-0 key over 50-60"], 61)
        self.assertEqual(tracked["picture with a prompt and one click"], 2)
        invalid = {row["name"]: row["reason"] for row in self.CASES["keyframes_invalid"]}
        self.assertEqual(invalid["range In 60 after Out 50"], "in_after_out")
        self.assertEqual(invalid["two keyframes at frame 10 for object 1"], "repeated_keyframe")

    def test_tracked_frames_rows(self):
        for row in self.CASES["tracked_frames"]:
            with self.subTest(row["name"]):
                self.assertEqual(tracked_frames(row["state"], row["frame_count"], row["prompt_set"]), row["tracked"])

    def test_nearest_keyframe_per_object_tie_goes_earlier(self):
        rows = {row["name"]: row for row in self.CASES["tracked_frames"]}
        row = rows["staggered keys over 40-120"]
        (first, last), layers = plan_tracking(row["state"], row["frame_count"], row["prompt_set"])
        self.assertEqual((first, last), (40, 120))
        spans = {obj: [(s.key, s.first, s.last) for s in layer] for obj, layer in layers}
        # A at 0 and 100: frame 50 is a tie, so it goes to 0; B at 50 takes the whole range.
        self.assertEqual(spans, {1: [(0, 40, 50), (100, 51, 120)], 2: [(50, 40, 120)]})

    def test_a_prompt_with_no_object_0_key_is_keyed_at_frame_0(self):
        state = json.dumps({"source": "a" * 64, "keyframes": [], "range": {"in": 50, "out": 60}})
        _, layers = plan_tracking(state, 150, True)
        (obj, layer), = layers
        self.assertEqual(obj, 0)
        (span,) = layer
        self.assertEqual((span.key, span.items, span.first, span.last), (0, None, 50, 60))
        self.assertEqual(span.calls, ((0, 60),))

    def test_steps_per_span(self):
        from wwai_markers_core import track_calls

        self.assertEqual(track_calls(5, 5, 5), ((5, 5),))
        self.assertEqual(track_calls(5, 5, 9), ((5, 9),))
        self.assertEqual(track_calls(5, 2, 5), ((5, 2),))
        self.assertEqual(track_calls(5, 2, 9), ((5, 9), (5, 2)))
        self.assertEqual(track_calls(0, 40, 50), ((0, 50),))
        self.assertEqual(track_calls(100, 51, 120), ((100, 120), (100, 51)))

    def test_parse_alone_checks_shape_order_and_repeats(self):
        state = json.dumps({"source": "a" * 64, "keyframes": [{"frame": 3, "object": 1, "items": []}]})
        keyframes, frame_range = parse_keyframes(state)
        self.assertEqual(frame_range, None)
        self.assertEqual([(k.frame, k.object) for k in keyframes], [(3, 1)])


def _ripple(**changes):
    """A valid v2 ripple manifest as text, with *changes* applied."""
    manifest = {
        "version": 2,
        "video": f"wwai/wwai-{'a' * 64}.mp4",
        "time_base": "1/12288",
        "window_start": 3,
        "length": 22,
        "decode_width": 64,
        "decode_height": 36,
        "megapixels": 0.25,
        "allow_upscale": False,
        "resize_method": "lanczos",
        "output_width": 0,
        "output_height": 0,
        "markers": [
            {"frame": 0, "pts": 1536, "edited": f"wwai/wwai-{'b' * 64}.png"},
            {"frame": 21, "pts": 12288, "edited": f"wwai/wwai-{'c' * 64}.png"},
        ],
    }
    manifest.update(changes)
    return json.dumps(manifest)


class RippleManifestTest(unittest.TestCase):
    def _refused(self, text):
        with self.assertRaises(MarkerContractError) as caught:
            parse_ripple_manifest(text)
        self.assertTrue(str(caught.exception).startswith("WWAI_RIPPLE_JOB_INVALID: "))

    def test_a_valid_manifest_parses(self):
        manifest = parse_ripple_manifest(_ripple())
        self.assertEqual((manifest["length"], manifest["window_start"], manifest["time_base"]), (22, 3, "1/12288"))
        self.assertEqual([(m["frame"], m["pts"]) for m in manifest["markers"]], [(0, 1536), (21, 12288)])

    def test_a_version_1_job_is_refused(self):
        v1 = {
            "version": 1, "clip": f"wwai/wwai-{'a' * 64}.mp4", "length": 22, "megapixels": 0.25,
            "allow_upscale": False, "resize_method": "lanczos", "output_width": 0, "output_height": 0,
            "markers": [{"frame": 0, "edited": f"wwai/wwai-{'b' * 64}.png"}],
        }
        self._refused(json.dumps(v1))
        self._refused(_ripple(version=1))

    def test_text_over_16_kib_is_refused_before_parsing(self):
        with mock.patch.object(wwai_markers_core.json, "loads") as loads:
            self._refused(" " * (16 * 1024 + 1))
        loads.assert_not_called()

    def test_a_bad_name_is_refused(self):
        self._refused(_ripple(video="wwai/wwai-short.mp4"))
        self._refused(_ripple(markers=[{"frame": 0, "pts": 0, "edited": "../../etc/passwd"}]))

    def test_a_frame_outside_the_length_is_refused(self):
        self._refused(_ripple(markers=[{"frame": 22, "pts": 0, "edited": f"wwai/wwai-{'b' * 64}.png"}]))

    def test_the_33rd_edit_is_refused_and_the_32nd_is_not(self):
        def edits(count):
            return [{"frame": i, "pts": i * 512, "edited": f"wwai/wwai-{'b' * 64}.png"} for i in range(count)]

        parse_ripple_manifest(_ripple(length=39, markers=edits(32)))
        self._refused(_ripple(length=39, markers=edits(33)))

    def test_anything_else_outside_the_schema_is_refused(self):
        edit = {"frame": 0, "pts": 0, "edited": f"wwai/wwai-{'b' * 64}.png"}
        for bad in (
            "not json",
            "[]",
            _ripple(version=3),
            _ripple(version=True),
            _ripple(time_base="1/0"),
            _ripple(time_base="0.5"),
            _ripple(time_base=24),
            _ripple(window_start=-1),
            _ripple(window_start=10_000_001),
            _ripple(length=23),  # not 17k + 5
            _ripple(length=3593),
            _ripple(decode_width=0),
            _ripple(decode_height=16385),
            _ripple(megapixels=0),
            _ripple(megapixels=True),
            _ripple(allow_upscale="no"),
            _ripple(resize_method="spline"),
            _ripple(output_width=512),  # width without height
            _ripple(output_width=16385, output_height=512),
            _ripple(markers=[]),
            _ripple(markers=[edit, edit]),  # repeated frame
            _ripple(markers=[edit, {**edit, "frame": 1}]),  # repeated pts
            _ripple(markers=[{**edit, "frame": 1.5}]),
            _ripple(markers=[{**edit, "pts": 1.5}]),
            _ripple(markers=[{**edit, "pts": True}]),
            _ripple(markers=[{"frame": 0, "edited": edit["edited"]}]),  # no pts
            _ripple(markers=[{**edit, "x": 1}]),
        ):
            with self.subTest(bad[:80]):
                self._refused(bad)
        extra = json.loads(_ripple())
        extra["surprise"] = 1
        self._refused(json.dumps(extra))


class RippleGridTest(unittest.TestCase):
    """The frame rule, from ripple_cases.json (byte-identical to WWAI's comfyui_ripple_cases.json)."""

    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ripple_cases.json"),
        encoding="utf-8",
    ) as _cases_file:
        CASES = json.load(_cases_file)

    def test_each_frame_lands_on_its_grid_frame(self):
        for row in self.CASES["grid"]:
            with self.subTest(row["name"]):
                tb = Fraction(row["time_base"])
                ends = row["pts"][1:] + [row["end_pts"]]
                got = [ripple_grid_frame(p, n, row["p0"], tb) for p, n in zip(row["pts"], ends)]
                self.assertEqual(got, row["frames"])
                self.assertEqual(ripple_grid_count(row["end_pts"], row["p0"], tb), row["count"])

    def test_the_slots_agree_with_the_frames_and_the_count(self):
        for row in self.CASES["grid"]:
            with self.subTest(row["name"]):
                # Packets come in decode order: the rule sorts them itself.
                slots = ripple_grid_slots(list(reversed(row["pts"])), row["p0"], row["end_pts"], Fraction(row["time_base"]))
                self.assertEqual(len(slots), row["count"])
                for pts, frame in zip(row["pts"], row["frames"]):
                    if frame is None:
                        self.assertNotIn(pts, slots)
                    else:
                        self.assertEqual(slots.index(pts), frame)

    def test_stamps_the_rule_cannot_place_are_refused(self):
        tb = Fraction(1, 12288)
        for pts, p0, end in (([0, 512, 512], 0, 1024), ([0, 512], 0, 512), ([], 0, 512), ([0, 512], 256, 1024)):
            with self.subTest(pts=pts, p0=p0, end=end):
                with self.assertRaisesRegex(MarkerContractError, "^WWAI_RIPPLE_VIDEO_UNUSABLE: "):
                    ripple_grid_slots(pts, p0, end, tb)


class RequireStoragePathTest(unittest.TestCase):
    def test_a_path_under_the_root_comes_back_unchanged(self):
        path = "/storenext/show/sh010/plate.%04d.png"
        self.assertIs(require_storage_path(path, "N", "source path"), path)

    def test_refusals(self):
        for path in (
            "plate.png", "./storenext/a.png", "", "   ", None, 5, "/storenext", "/storenextX/a.png",
            "/opt/x.png", "/storenext//a.png", "/storenext/a/", "/storenext/", "/storenext/./a.png",
            "/storenext/a/../b.png", "/storenext/a\\b.png", "/storenext/a\x00b.png",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(MarkerContractError, r"^WWAI_PATH_REFUSED: Cls: output path ") as caught:
                    require_storage_path(path, "Cls", "output path")
                self.assertIn(repr(path), str(caught.exception))

    def test_the_root_is_read_at_call_time(self):
        with mock.patch.object(wwai_markers_core, "STORAGE_ROOT", "/tmp/x/"):
            self.assertEqual(require_storage_path("/tmp/x/a.png", "N", "source path"), "/tmp/x/a.png")
            with self.assertRaises(MarkerContractError):
                require_storage_path("/storenext/a.png", "N", "source path")


class FramePatternTest(unittest.TestCase):
    def test_none_one_two_and_percent_in_the_folder(self):
        self.assertIsNone(frame_pattern("/storenext/a/plate.png"))
        self.assertEqual(frame_pattern("/storenext/a/plate.%04d.png"), ("/storenext/a", "plate.", 4, ".png"))
        self.assertEqual(frame_pattern("/storenext/a/%01d"), ("/storenext/a", "", 1, ""))
        for path in ("/storenext/a/p.%04d.%04d.png", "/storenext/100%/p.%04d.png", "/storenext/a/p.%4d.png", "/storenext/a/p%.png", "/storenext/a/p.%04d%.png"):
            with self.subTest(path=path):
                with self.assertRaisesRegex(MarkerContractError, "^WWAI_PATH_REFUSED: "):
                    frame_pattern(path)


class SequenceFramesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def touch(self, *names):
        for name in names:
            open(os.path.join(self.dir, name), "wb").close()

    def test_only_exact_width_matches_come_back_sorted(self):
        self.touch("p.0010.png", "p.0002.png", "p.002.png", "p.00003.png", "p.0004.jpg", "q.0005.png", "p.000a.png")
        frames = sequence_frames(f"{self.dir}/p.%04d.png", "Cls")
        self.assertEqual(frames, [(2, f"{self.dir}/p.0002.png"), (10, f"{self.dir}/p.0010.png")])

    def test_no_match_is_refused(self):
        self.touch("other.png")
        with self.assertRaisesRegex(MarkerContractError, "^WWAI_SEQUENCE_EMPTY: Cls"):
            sequence_frames(f"{self.dir}/p.%04d.png", "Cls")


class WriteInPlaceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.final = os.path.join(self.dir, "out.png")

    @staticmethod
    def put(data):
        def write(temp_path):
            with open(temp_path, "wb") as handle:
                handle.write(data)

        return write

    def test_writes_and_returns_the_final_path_leaving_no_temp(self):
        self.assertEqual(write_in_place(self.final, self.put(b"one")), self.final)
        with open(self.final, "rb") as handle:
            self.assertEqual(handle.read(), b"one")
        self.assertEqual(os.listdir(self.dir), ["out.png"])

    def test_overwrite_is_allowed(self):
        write_in_place(self.final, self.put(b"one"))
        write_in_place(self.final, self.put(b"two"))
        with open(self.final, "rb") as handle:
            self.assertEqual(handle.read(), b"two")

    def test_a_failing_write_raises_leaves_no_temp_and_keeps_the_old_file(self):
        write_in_place(self.final, self.put(b"old"))

        def broken(temp_path):
            with open(temp_path, "wb") as handle:
                handle.write(b"half")
            raise OSError("disk full")

        with self.assertRaisesRegex(OSError, "disk full"):
            write_in_place(self.final, broken)
        self.assertEqual(os.listdir(self.dir), ["out.png"])
        with open(self.final, "rb") as handle:
            self.assertEqual(handle.read(), b"old")

    def test_no_path_comes_back_when_the_rename_fails(self):
        with mock.patch.object(os, "replace", side_effect=OSError("rename refused")):
            with self.assertRaisesRegex(OSError, "rename refused"):
                write_in_place(self.final, self.put(b"data"))
        self.assertEqual(os.listdir(self.dir), [])

    def test_a_missing_output_folder_fails_loud_and_is_not_created(self):
        missing = os.path.join(self.dir, "nope", "out.png")
        with self.assertRaises(FileNotFoundError):
            write_in_place(missing, self.put(b"data"))
        self.assertFalse(os.path.exists(os.path.dirname(missing)))


if __name__ == "__main__":
    unittest.main()
