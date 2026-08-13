from argparse import ArgumentParser, Namespace
from pathlib import Path
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Inspect ResNet18 preprocessing for one image."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional directory for preprocessing inspection images.",
    )
    return parser.parse_args()


def save_inspection_images(
    output_dir: Path,
    image_path: Path,
    original_image: Image.Image,
    resized_image: Image.Image,
    cropped_image: Image.Image,
    direct_resized_image: Image.Image,
) -> None:
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    original_image.save(resolved_output_dir / f"{stem}_original.png")
    resized_image.save(resolved_output_dir / f"{stem}_resized.png")
    cropped_image.save(resolved_output_dir / f"{stem}_cropped.png")
    direct_resized_image.save(
        resolved_output_dir / f"{stem}_direct_224.png"
    )


def main() -> None:
    arguments = parse_arguments()
    image_path = arguments.image.resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    weights = ResNet18_Weights.DEFAULT
    preprocessing = weights.transforms()

    resize = Resize(
        size=preprocessing.resize_size,
        interpolation=preprocessing.interpolation,
        antialias=True,
    )
    center_crop = CenterCrop(size=preprocessing.crop_size)

    direct_resize = Resize(
        size=(224, 224),
        interpolation=preprocessing.interpolation,
        antialias=True,
    )

    bottle_preprocessing = Compose(
        [
            direct_resize,
            ToTensor(),
            Normalize(
                mean=preprocessing.mean,
                std=preprocessing.std,
            ),
        ]
    )

    resize = Resize(
        size=preprocessing.resize_size,
        interpolation=preprocessing.interpolation,
        antialias=True,
    )
    center_crop = CenterCrop(size=preprocessing.crop_size)

    direct_resize = Resize(
        size=preprocessing.crop_size,
        interpolation=preprocessing.interpolation,
        antialias=True,
    )

    with Image.open(image_path) as source_image:
        rgb_image = source_image.convert("RGB")
        resized_image = resize(rgb_image)
        cropped_image = center_crop(resized_image)
        direct_resized_image = direct_resize(rgb_image)
        tensor = bottle_preprocessing(rgb_image)

        if arguments.output_dir is not None:
            save_inspection_images(
                arguments.output_dir,
                image_path,
                rgb_image,
                resized_image,
                cropped_image,
                direct_resized_image,
            )

        print(f"Image: {image_path}")
        print(f"Original mode: {source_image.mode}")
        print(f"Converted mode: {rgb_image.mode}")
        print(f"Original size: {source_image.size}")

    print(f"Tensor shape: {tuple(tensor.shape)}")
    print(f"Tensor data type: {tensor.dtype}")
    print(f"Minimum value: {tensor.min().item():.4f}")
    print(f"Maximum value: {tensor.max().item():.4f}")
    print("Bottle preprocessing configuration:")
    print(bottle_preprocessing)
    if arguments.output_dir is not None:
        print(f"Inspection images: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()