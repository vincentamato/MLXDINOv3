# Changelog

## Unreleased

### Changed

- `DinoVisionTransformer.loadPretrained(from:)` now checks the weights against the model that
  `config.json` describes and throws a `DinoVisionTransformer.LoadError` when a tensor is
  missing, surplus or of the wrong shape, when the layer count differs, or when the checkpoint
  is still in PyTorch layout. Before, such files loaded and left part of the model randomly
  initialised or crashed on the first forward pass.
- The model casts its input to the precision of the weights, as the reference implementation
  does. A half-precision checkpoint therefore runs, and returns, half precision even when given
  the `float32` output of `ImageProcessor`. On a `bfloat16` ViT-H+/16 at 512 px this lowers peak
  memory from about 3.1 GB to 2.3 GB. To keep single-precision results, cast the weights once
  with `model.apply { $0.asType(.float32) }`.
- A malformed input (wrong rank, channels-first, smaller than one patch) stops with a message
  naming the expected `[batch, height, width, channels]` layout instead of an error from inside
  the convolution.
- `config.json` is validated: a `model_type` other than `dinov3_vit`, an unsupported
  `hidden_act`, or a `hidden_size` that the heads cannot divide throws instead of trapping or
  being ignored.
- `ImageProcessor` decodes large images in bands and no longer makes an extra full-size copy.
  A 48-megapixel photo adds about 310 MB at peak instead of about 620 MB. On iOS it reads the
  `UIImage` bitmap directly rather than re-rendering it first.
- The `Convert` tool validates `config.json` before downloading weights, recognises tensor
  names with a prefix, explains how to get access when a model is gated, and prints the
  correct command name in its usage text.

### Added

- `DinoVisionTransformer.patchSize`, `inChannels`, `embedDim`, `depth`, `numHeads` and
  `numRegisterTokens` are public.
- `hidden_act` in `config.json` is honoured (`gelu`, `silu`, `relu`), and the initialiser takes
  an optional `activation`. Before, the gated feed-forward layer always used SiLU and the plain
  one always used GELU.
- `ImageProcessor` accepts a `CGImage` with an orientation, and a `colorProfile` option:
  `.convertToSRGB` (the default, and the previous behaviour) or `.ignore`, which uses the stored
  values the way Pillow and torchvision do.
- `Convert` accepts a local directory in place of a Hugging Face model ID, and
  `--strip-prefix` to extract the backbone of a fine-tuned checkpoint.
- Checkpoints converted by 1.x, which kept `embeddings.mask_token`, load again.
- Tests that need no download: the forward pass against Hugging Face's PyTorch implementation
  on random-weight models (plain and gated feed-forward layers, with and without register
  tokens, batches, non-square and non-patch-multiple inputs, half precision), loader rejection
  cases, `ImageProcessor` against Pillow, and the converter. `Scripts/make_test_fixtures.py`
  regenerates the fixtures.
- Documentation comments on the public API, `THIRD_PARTY_NOTICES.md`, and a CI workflow.

### Fixed

- The rotary-embedding frequencies were registered as a trainable parameter
  (`ropeEmbed.invFreq`): an optimizer would have updated them, and they were written into saved
  checkpoints. They are now a fixed constant; a checkpoint that was saved with that key still loads.
- `PlatformImage` was declared twice when building for Mac Catalyst.
- The test image is loaded by its real file name, so the tests no longer depend on a
  case-insensitive file system.

## 2.0.0

Major refactor with a smaller public API.

- `loadPretrained(modelPath:)` became `DinoVisionTransformer.loadPretrained(from:)`.
- `DinoOutput`: `poolerOutput` became `clsToken`; `patchTokens` was added; `hiddenStates` is
  `[MLXArray]?` instead of `[String: MLXArray]?`; `attentions` and the deprecated properties
  were removed.
- `pos_embed_rescale` is no longer applied at inference. 1.x multiplied every patch position by
  the value in `config.json` (2.0 in the released checkpoints), which is a training-time
  augmentation in the reference implementation. Outputs therefore differ from 1.x, and match
  PyTorch.
- `forwardFeatures`, `embedDim`, `PreprocessError` and the individual layer classes are no
  longer public.
- The converter drops `embeddings.mask_token`, and 2.0.0 rejects checkpoints that still have
  it, so models converted by 1.x need converting again (later versions accept them).
- Requires mlx-swift 0.31.3 or later (1.0.0 required 0.29.1).

## 1.0.0

Initial release.
