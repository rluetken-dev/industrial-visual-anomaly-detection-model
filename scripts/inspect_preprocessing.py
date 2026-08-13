from argparse import ArgumentParser, Namespace
from pathlib import Path

from PIL import Image
from torchvision.models import ResNet18_Weights


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Inspect ResNet18 preprocessing for one image."
    )
    parser.add_argument("--image", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    image_path = arguments.image.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    weights = ResNet18_Weights.DEFAULT
    preprocessing = weights.transforms()

    with Image.open(image_path) as source_image:
        rgb_image = source_image.convert("RGB")
        tensor = preprocessing(rgb_image)

        print(f"Image: {image_path}")
        print(f"Original mode: {source_image.mode}")
        print(f"Converted mode: {rgb_image.mode}")
        print(f"Original size: {source_image.size}")

    print(f"Tensor shape: {tuple(tensor.shape)}")
    print(f"Tensor data type: {tensor.dtype}")
    print(f"Minimum value: {tensor.min().item():.4f}")
    print(f"Maximum value: {tensor.max().item():.4f}")
    print("Preprocessing configuration:")
    print(preprocessing)


if __name__ == "__main__":
    main()