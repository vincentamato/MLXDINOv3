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

The `Convert` target converts a Hugging Face checkpoint to MLX format. Only ViT models are supported.

```bash
./mlx-run.sh Convert facebook/dinov3-vits16-pretrain-lvd1689m ./Models/dinov3-vits16-mlx
```

Meta's checkpoints are gated. Accept the license on the model page, create an [access token](https://huggingface.co/settings/tokens), and run the command with `HF_TOKEN=<your token>` in the environment.

The first argument can also be a local directory holding `config.json` and `model.safetensors`. For a fine-tuned checkpoint whose backbone tensors carry a prefix, `--strip-prefix` extracts the backbone:

```bash
./mlx-run.sh Convert ./my-finetuned-checkpoint ./Models/my-backbone-mlx --strip-prefix backbone.
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

`loadPretrained` checks the weights against `config.json` and throws a `DinoVisionTransformer.LoadError` that says what does not match.

### Inputs

The model takes a channels-last batch, `[batch, height, width, 3]`. Height and width can differ and need not be multiples of the patch size.

`ImageProcessor` produces that tensor from a `UIImage`, an `NSImage` or a `CGImage`:

- It resizes to `size` x `size` (224 by default) by stretching, without preserving the aspect ratio, like the Hugging Face DINOv3 processor. Pass `ImageProcessor(size: 512)` for another resolution.
- The image's orientation is applied, so the model sees it upright.
- Transparent pixels are composited onto black.
- Wide-gamut images, such as Display P3 photos from an iPhone, are converted to sRGB. Pillow and torchvision use the stored values unchanged instead; pass `colorProfile: .ignore` to reproduce a Python pipeline.
- Normalization defaults to the ImageNet statistics. Pass `normalizationStats:` for a checkpoint trained with other statistics.
- Resizing matches Pillow's bilinear filter bit for bit (Pillow 9.5 to 12.x).

### Outputs

- `lastHiddenState`: every token after the final layer norm, in the order CLS, register tokens, patch tokens.
- `clsToken` and `patchTokens`: the corresponding slices of `lastHiddenState`. Patch tokens are in row-major order.
- `hiddenStates` (with `outputHiddenStates: true`): the embedding output followed by the output of each block, without the final layer norm.

### Precision and memory

The model runs in the precision of its weights, so a `float16` or `bfloat16` checkpoint uses about half the memory of a `float32` one and returns half-precision features. Half-precision features can differ from single-precision ones by a few percent. To run a half-precision checkpoint in single precision, cast the weights once after loading:

```swift
model.apply { $0.asType(.float32) }
```

`ImageProcessor` is much slower in a Debug build (seconds for a 12-megapixel photo, against about 0.1 s in Release), so measure in Release.

## Testing

Tests use `xcodebuild` because MLX depends on the Metal backend (`swift test` won't work).

```bash
xcodebuild test -scheme MLXDINOv3-Package -destination 'platform=macOS' -derivedDataPath .build/xcode
```

Most tests need no download. They compare the model with Hugging Face's PyTorch implementation on small random-weight models, compare `ImageProcessor` with Pillow, and cover the loader and the converter, using the files in `Tests/MLXDINOv3Tests/Fixtures`. To regenerate those files with [uv](https://docs.astral.sh/uv/):

```bash
uv run Scripts/make_test_fixtures.py
```

One more suite checks the real `dinov3-vits16-pretrain-lvd1689m` checkpoint end to end against PyTorch reference outputs downloaded from Hugging Face. It is skipped until the converted model is in the test resources directory:

```bash
./mlx-run.sh Convert facebook/dinov3-vits16-pretrain-lvd1689m Tests/MLXDINOv3Tests/Resources
```

## References

- [DINOv3 paper](https://arxiv.org/abs/2508.10104)
- [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## License

MIT. See [LICENSE](LICENSE). The image resizing code is a port of Pillow's; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Pretrained weights are under Meta's [DINOv3 License](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
