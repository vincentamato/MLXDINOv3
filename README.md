# MLXDINOv3

A native Swift implementation of Meta’s [DINOv3](https://arxiv.org/abs/2508.10104) using [MLX Swift](https://github.com/ml-explore/mlx-swift).

DINOv3 is a family of self-supervised vision foundation models from [Meta AI](https://ai.meta.com/), producing high-quality dense visual features that outperform specialized models without fine-tuning. This package provides a numerically validated, on-device compatible version for Apple silicon.

## Installation

Add MLXDINOv3 to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/vincentamato/MLXDINOv3.git", from: "1.0.0")
]
```

Then import it:

```swift
import MLXDINOv3
```

## Converting Hugging Face weights to MLX format

Convert Hugging Face weights to MLX format using the conversion CLI in Xcode:

1. Open the package in Xcode: `xed .`
2. Select the `Convert` scheme from the scheme selector
3. Edit the scheme (Product → Scheme → Edit Scheme)
4. Under "Run" → "Arguments", add:
   - `facebook/dinov3-vits16-pretrain-lvd1689m`
   - `./Models/dinov3-vits16-mlx`
5. Run the scheme (Cmd+R)

> **Note**: Currently, only the ViT models are supported.

## Example Usage

```swift
import AppKit
import MLX
import MLXDINOv3

// Load a pretrained model
let model = try loadPretrained(modelPath: "Models/dinov3-vits16-mlx")

// Preprocess an image
let image = NSImage(contentsOfFile: "image.jpg")!
let processor = ImageProcessor()
let inputs = try processor(image)

// Run inference
let outputs = model(inputs)

print("Pooler output shape:", outputs.poolerOutput.shape)
print("Last hidden state shape:", outputs.lastHiddenState.shape)
```

## Testing

All testing must be done from Xcode due to MLX metallib requirements.

**Step 1: Convert the test model**

1. Open the package in Xcode: `xed .`
2. Select the `Convert` scheme
3. Edit the scheme (Product → Scheme → Edit Scheme)
4. Under "Run" → "Arguments", add:
   - `facebook/dinov3-vits16-pretrain-lvd1689m`
   - `Tests/MLXDINOv3Tests/Resources/Model`
5. Run (Cmd+R)

**Step 2: Run tests**

Run tests with **Cmd+U** or **Product → Test**

Tests automatically download PyTorch reference outputs from HuggingFace Hub for validation.

## References

- DINOv3 Paper: [DINOv3](https://arxiv.org/abs/2508.10104)
- DINOv3 Repository: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## License

This package is released under the [MIT License](LICENSE).

Note: The pretrained DINOv3 weights and original model architecture are released under Meta’s [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
