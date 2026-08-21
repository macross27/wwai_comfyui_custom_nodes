"""Tests for the rules the marker nodes cannot be allowed to get wrong.

Runs anywhere: `python -m unittest` from the repo root, no ComfyUI, no torch.
"""

import os
import shutil
import tempfile
import unittest

from wwai_markers_core import (
    MarkerContractError,
    copy_file_verbatim,
    extension_of,
    require_marker_name,
    require_single,
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


if __name__ == "__main__":
    unittest.main()
