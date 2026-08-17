from base64 import b64decode
from io import BytesIO
import unittest

import torch
from PIL import Image

from industrial_visual_anomaly_detection.service.heatmap_encoding import (
    encode_heatmap_png_base64,
)


class HeatmapEncodingTests(unittest.TestCase):
    def test_patch_scores_are_encoded_as_rgb_png(self) -> None:
        encoded_heatmap = encode_heatmap_png_base64(
            patch_scores=torch.tensor(
                [
                    [0.0, 2.0],
                    [4.0, 8.0],
                ]
            ),
            threshold=4.0,
            output_size=(8, 12),
        )

        image_bytes = b64decode(encoded_heatmap)

        with Image.open(BytesIO(image_bytes)) as heatmap:
            self.assertEqual("PNG", heatmap.format)
            self.assertEqual("RGB", heatmap.mode)
            self.assertEqual((12, 8), heatmap.size)

    def test_non_positive_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            encode_heatmap_png_base64(
                patch_scores=torch.zeros(2, 2),
                threshold=0.0,
                output_size=(8, 8),
            )


if __name__ == "__main__":
    unittest.main()