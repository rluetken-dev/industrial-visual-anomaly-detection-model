import unittest

from PIL import Image

from industrial_visual_anomaly_detection.preprocessing import (
    create_bottle_preprocessing,
    create_image_preprocessing,
)


class PreprocessingTests(unittest.TestCase):
    def test_custom_input_size_is_applied(self) -> None:
        preprocessing = create_image_preprocessing(
            input_size=(320, 320)
        )

        tensor = preprocessing(
            Image.new("RGB", (1000, 1000))
        )

        self.assertEqual((3, 320, 320), tuple(tensor.shape))

    def test_bottle_wrapper_retains_default_size(self) -> None:
        preprocessing = create_bottle_preprocessing()

        tensor = preprocessing(
            Image.new("RGB", (900, 900))
        )

        self.assertEqual((3, 224, 224), tuple(tensor.shape))

    def test_invalid_input_size_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "greater than zero",
        ):
            create_image_preprocessing(
                input_size=(0, 320)
            )


if __name__ == "__main__":
    unittest.main()