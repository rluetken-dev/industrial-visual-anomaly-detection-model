from torchvision.transforms import (
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_DEVIATION = [0.229, 0.224, 0.225]
BOTTLE_INPUT_SIZE = (224, 224)


def create_bottle_preprocessing() -> Compose:
    """Create the deterministic preprocessing for the first Bottle baseline."""

    return Compose(
        [
            Resize(
                size=BOTTLE_INPUT_SIZE,
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