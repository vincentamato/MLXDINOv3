# MLXDINOv3

Swift port of Meta's [DINOv3](https://arxiv.org/abs/2508.10104) using [MLX Swift](https://github.com/ml-explore/mlx-swift).

DINOv3 is a vision foundation model that produces dense visual features useful for classification, segmentation, retrieval, etc. without fine-tuning. This package implements the ViT architecture in MLX and validates outputs against a PyTorch reference.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/vincentamato/MLXDINOv3.git", from: "2.0.0")
]
```

## Converting Weights

The `Convert` target downloads a Hugging Face checkpoint and converts it to MLX format. Only ViT models are supported.

```bash
./mlx-run.sh Convert facebook/dinov3-vits16-pretrain-lvd1689m ./Models/dinov3-vits16-mlx
```

## Usage

```swift
import AppKit
import MLX
import MLXDINOv3

let model = try DinoVisionTransformer.loadPretrained(from: "Models/dinov3-vits16-mlx")

let image = NSImage(contentsOfFile: "image.jpg")!
let inputs = try ImageProcessor()(image)

let outputs = model(inputs)

print("CLS token shape:", outputs.clsToken.shape)
print("Patch tokens shape:", outputs.patchTokens.shape)
print("Last hidden state shape:", outputs.lastHiddenState.shape)
```

## Testing

Tests use `xcodebuild` because MLX depends on the Metal backend (`swift test` won't work). Before running tests, convert the `dinov3-vits16-pretrain-lvd1689m` model, making sure it is saved to the test resources directory.

```bash
# Convert the model
./mlx-run.sh Convert facebook/dinov3-vits16-pretrain-lvd1689m Tests/MLXDINOv3Tests/Resources

# Run tests
xcodebuild test -scheme MLXDINOv3-Package -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Tests download PyTorch reference outputs from Hugging Face and compare against them using cosine similarity and relative L2 error.

## References

- [DINOv3 paper](https://arxiv.org/abs/2508.10104)
- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## License

MIT. See [LICENSE](LICENSE).

Pretrained weights are under Meta's [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
