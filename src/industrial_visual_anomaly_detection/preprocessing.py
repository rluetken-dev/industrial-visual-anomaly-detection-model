from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_DEVIATION = [0.229, 0.224, 0.225]
DEFAULT_INPUT_SIZE = (224, 224)
BOTTLE_INPUT_SIZE = DEFAULT_INPUT_SIZE


def create_image_preprocessing(
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> Compose:
    """Create deterministic ImageNet-compatible preprocessing."""

    if input_size[0] <= 0 or input_size[1] <= 0:
        raise ValueError(
            "Input dimensions must be greater than zero."
        )

    return Compose(
        [
            Resize(
                size=input_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            ToTensor(),
            Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STANDARD_DEVIATION,
            ),
        ]
    )


def create_bottle_preprocessing() -> Compose:
    """Create the preprocessing retained for Bottle compatibility."""

    return create_image_preprocessing(
        input_size=BOTTLE_INPUT_SIZE,
    )