import time
import onnx
import onnxruntime

import torch.nn.functional as functional

import torch
from torchvision.models import ResNet18_Weights, resnet18

class ResNet18FeatureExtractor(torch.nn.Module):
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

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()

feature_extractor = ResNet18FeatureExtractor(model)
feature_extractor.eval()

captured_features: dict[str, torch.Tensor] = {}


def capture_output(name: str):
    def hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured_features[name] = output

    return hook

hooks = [
    model.layer2.register_forward_hook(capture_output("layer2")),
    model.layer3.register_forward_hook(capture_output("layer3")),
]

image_batch = torch.rand(1, 3, 224, 224)

with torch.inference_mode():
    for _ in range(3):
        model(image_batch)

    start_time = time.perf_counter()

    for _ in range(20):
        model(image_batch)

    elapsed_seconds = time.perf_counter() - start_time

layer2_features = captured_features["layer2"]
layer3_features = captured_features["layer3"]

resized_layer3_features = functional.interpolate(
    layer3_features,
    size=layer2_features.shape[-2:],
    mode="bilinear",
    align_corners=False,
)

combined_features = torch.cat(
    [layer2_features, resized_layer3_features],
    dim=1,
)

patch_embeddings = (
    combined_features
    .permute(0, 2, 3, 1)
    .reshape(-1, combined_features.shape[1])
)

for hook in hooks:
    hook.remove()

print(f"Input shape: {tuple(image_batch.shape)}")
print(f"Layer 2 feature shape: {tuple(captured_features['layer2'].shape)}")
print(f"Layer 3 feature shape: {tuple(captured_features['layer3'].shape)}")

average_milliseconds = elapsed_seconds / 20 * 1000

print(f"Average inference time: {average_milliseconds:.2f} ms")

print(f"Combined feature shape: {tuple(combined_features.shape)}")
print(f"Patch embedding shape: {tuple(patch_embeddings.shape)}")

onnx_path = "resnet18_feature_extractor.onnx"

torch.onnx.export(
    feature_extractor,
    (image_batch,),
    onnx_path,
    input_names=["image"],
    output_names=["layer2_features", "layer3_features"],
    opset_version=18,
)

print(f"ONNX model exported to: {onnx_path}")

onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print("ONNX model validation succeeded.")

session = onnxruntime.InferenceSession(
    onnx_path,
    providers=["CPUExecutionProvider"],
)

onnx_layer2, onnx_layer3 = session.run(
    None,
    {"image": image_batch.numpy()},
)

with torch.inference_mode():
    pytorch_layer2, pytorch_layer3 = feature_extractor(image_batch)

layer2_max_error = (
    pytorch_layer2 - torch.from_numpy(onnx_layer2)
).abs().max().item()

layer3_max_error = (
    pytorch_layer3 - torch.from_numpy(onnx_layer3)
).abs().max().item()

print(f"Layer 2 maximum difference: {layer2_max_error:.8f}")
print(f"Layer 3 maximum difference: {layer3_max_error:.8f}")