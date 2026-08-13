import torch
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18FeatureExtractor(torch.nn.Module):
    """Extract intermediate ResNet18 feature maps for anomaly detection."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone.conv1(image)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        layer2_features = self.backbone.layer2(x)
        layer3_features = self.backbone.layer3(layer2_features)

        return layer2_features, layer3_features


def create_resnet18_feature_extractor() -> ResNet18FeatureExtractor:
    """Create the frozen pretrained feature extractor for the first baseline."""

    backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

    for parameter in backbone.parameters():
        parameter.requires_grad = False

    feature_extractor = ResNet18FeatureExtractor(backbone)
    feature_extractor.eval()

    return feature_extractor