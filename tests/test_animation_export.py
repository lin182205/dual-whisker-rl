from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

from matplotlib import animation
from matplotlib import pyplot as plt

from dual_whisker_rl.visualization import save_matplotlib_animation


def make_test_animation(*, figsize: tuple[float, float] = (2.0, 2.0)):
    fig, ax = plt.subplots(figsize=figsize)
    (line,) = ax.plot([], [])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def update(frame: int):
        value = frame / 2.0
        line.set_data([0.0, value], [0.0, value])
        return (line,)

    return fig, animation.FuncAnimation(fig, update, frames=3, blit=False)


class AnimationExportTests(unittest.TestCase):
    @unittest.skipUnless(
        animation.writers.is_available("ffmpeg"),
        "FFmpeg is required for MP4 export",
    )
    def test_mp4_export_pads_odd_dimensions_and_uses_h264_yuv420p(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "odd.mp4"
            fig, anim = make_test_animation(figsize=(1.0, 1.0))
            try:
                save_matplotlib_animation(anim, path, fps=4, dpi=101)
            finally:
                plt.close(fig)

            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            ffprobe = shutil.which("ffprobe")
            if ffprobe is None:
                return
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,pix_fmt,width,height,r_frame_rate",
                    "-of",
                    "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual(stream["codec_name"], "h264")
            self.assertEqual(stream["pix_fmt"], "yuv420p")
            self.assertEqual(int(stream["width"]) % 2, 0)
            self.assertEqual(int(stream["height"]) % 2, 0)
            self.assertEqual(stream["r_frame_rate"], "4/1")

    def test_explicit_gif_export_remains_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "preview.gif"
            fig, anim = make_test_animation()
            try:
                save_matplotlib_animation(anim, path, fps=4, dpi=80)
            finally:
                plt.close(fig)

            self.assertTrue(path.is_file())
            self.assertTrue(path.read_bytes().startswith(b"GIF"))

    def test_unsupported_extension_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "只支持 .mp4 和 .gif"):
            save_matplotlib_animation(mock.Mock(), Path("animation.avi"), fps=4)

    def test_missing_ffmpeg_has_actionable_error(self) -> None:
        with mock.patch.object(
            animation.writers,
            "is_available",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "将 ffmpeg 加入 PATH"):
                save_matplotlib_animation(mock.Mock(), Path("animation.mp4"), fps=4)


if __name__ == "__main__":
    unittest.main()
