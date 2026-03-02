# MLXDINOv3

Swift port of Meta's [DINOv3](https://arxiv.org/abs/2508.10104) on [MLX Swift](https://github.com/ml-explore/mlx-swift).

DINOv3 is a self-supervised vision model from Meta that produces dense visual features useful for classification, segmentation, retrieval, etc. without fine-tuning. This package implements the architecture in MLX and validates outputs against a PyTorch reference.

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/vincentamato/MLXDINOv3.git", from: "1.0.0")
]
```

```swift
import MLXDINOv3
```

## Converting weights

The `Convert` target downloads a Hugging Face checkpoint and converts it to MLX format. Only ViT models are supported for now.

```bash
xcodebuild build -scheme Convert -destination platform=macOS -derivedDataPath .build/xcode && \
  .build/xcode/Build/Products/Release/Convert \
    facebook/dinov3-vits16-pretrain-lvd1689m \
    ./Models/dinov3-vits16-mlx
```

## Usage

```swift
import AppKit
import MLX
import MLXDINOv3

let model = try loadPretrained(modelPath: "Models/dinov3-vits16-mlx")

let image = NSImage(contentsOfFile: "image.jpg")!
let processor = ImageProcessor()
let inputs = try processor(image)

let outputs = model(inputs)

print("Pooler output shape:", outputs.poolerOutput.shape)
print("Last hidden state shape:", outputs.lastHiddenState.shape)
```

## Testing

Tests use `xcodebuild` because MLX depends on the Metal backend (`swift test` won't work). Before running tests, you need to convert the model into the test resources directory.

```bash
# Convert the test model (skip if already done)
xcodebuild build -scheme Convert -destination platform=macOS -derivedDataPath .build/xcode && \
  .build/xcode/Build/Products/Release/Convert \
    facebook/dinov3-vits16-pretrain-lvd1689m \
    Tests/MLXDINOv3Tests/Resources/Model

# Run tests
xcodebuild test -scheme MLXDINOv3Tests -destination platform=macOS
```

Tests download PyTorch reference outputs from Hugging Face and compare against them.

## References

- [DINOv3 paper](https://arxiv.org/abs/2508.10104)
- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## License

MIT. See [LICENSE](LICENSE).

Pretrained weights are under Meta's [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
