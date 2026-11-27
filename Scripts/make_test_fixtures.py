# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#     "numpy",
#     "pillow==12.0.0",
#     "safetensors",
#     "torch",
#     "transformers==4.56.2",
# ]
# ///
"""Regenerates Tests/MLXDINOv3Tests/Fixtures.

    uv run Scripts/make_test_fixtures.py

Model fixtures are tiny DINOv3 ViTs with random weights, run through the Hugging Face
PyTorch implementation, so the Swift model can be checked against the reference without
downloading the gated checkpoints. Image fixtures hold what Pillow produces for a few
small images, to check ImageProcessor.

Pillow is pinned to 12.0.0 because ImageProcessor ports its resampler; Pillow 13 sometimes
runs the vertical pass first and can differ by one level.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.numpy import save_file as save_numpy
from safetensors.torch import save_file
from transformers import DINOv3ViTConfig, DINOv3ViTModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Tests/MLXDINOv3Tests/Fixtures"
PHOTO = ROOT / "Tests/MLXDINOv3Tests/Resources/Image.jpg"

# name -> (config overrides, {input name: NCHW shape})
MODELS = {
    # plain MLP; a batch of two non-square images, and a size that is not a multiple of the patch
    "plain": (
        dict(patch_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
             num_attention_heads=2, num_register_tokens=4, use_gated_mlp=False, hidden_act="gelu"),
        {"batch": (2, 3, 32, 48), "ragged": (1, 3, 38, 50)},
    ),
    # gated MLP, with every optional setting away from its default
    "gated": (
        dict(patch_size=8, hidden_size=48, intermediate_size=64, num_hidden_layers=2,
             num_attention_heads=3, num_register_tokens=4, use_gated_mlp=True, hidden_act="silu",
             key_bias=True, query_bias=False, value_bias=False, mlp_bias=False, proj_bias=False,
             layer_norm_eps=1e-3, rope_theta=10000.0, layerscale_value=0.1),
        {"single": (1, 3, 40, 24)},
    ),
    "no_registers": (
        dict(patch_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
             num_attention_heads=2, num_register_tokens=0, use_gated_mlp=False, hidden_act="gelu"),
        {"single": (1, 3, 16, 16)},
    ),
    # activations other than the pairing of the released checkpoints
    "gated_gelu": (
        dict(patch_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
             num_attention_heads=2, num_register_tokens=4, use_gated_mlp=True, hidden_act="gelu"),
        {"single": (1, 3, 16, 16)},
    ),
    "plain_relu": (
        dict(patch_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
             num_attention_heads=2, num_register_tokens=4, use_gated_mlp=False, hidden_act="relu"),
        {"single": (1, 3, 16, 16)},
    ),
}


def make_models():
    torch.manual_seed(0)
    for name, (overrides, inputs) in MODELS.items():
        config = DINOv3ViTConfig(**overrides)
        model = DINOv3ViTModel(config).eval()
        # The default initialisation leaves biases at zero and layer scale constant. Perturb
        # every tensor so that one that is ignored or swapped shows up in the output.
        with torch.no_grad():
            for key, p in model.named_parameters():
                if "lambda1" in key:
                    p.copy_(torch.empty_like(p).uniform_(0.5, 1.5) * config.layerscale_value)
                elif "norm" in key and key.endswith("weight"):
                    p.copy_(torch.empty_like(p).uniform_(0.5, 1.5))
                else:
                    p.add_(torch.randn_like(p) * 0.05)

        hidden = {}
        model.embeddings.register_forward_hook(lambda m, i, o: hidden.__setitem__(0, o))
        for index, layer in enumerate(model.layer):
            layer.register_forward_hook(lambda m, i, o, n=index + 1: hidden.__setitem__(n, o))

        io = {}
        for input_name, shape in inputs.items():
            pixel_values = torch.randn(*shape)
            with torch.no_grad():
                result = model(pixel_values)
            io[f"{input_name}.pixel_values"] = pixel_values.permute(0, 2, 3, 1).contiguous()
            io[f"{input_name}.last_hidden_state"] = result.last_hidden_state.clone()
            io[f"{input_name}.pooler_output"] = result.pooler_output.clone()
            for n, state in hidden.items():
                io[f"{input_name}.hidden_state_{n}"] = state.clone()

        out = OUT / "models" / name
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(config.to_json_string())
        save_file(io, str(out / "io.safetensors"))

        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        # What the Convert tool writes: no mask token, convolution weight channels-last.
        converted = {
            k: (v.permute(0, 2, 3, 1).contiguous() if k == "embeddings.patch_embeddings.weight" else v)
            for k, v in state.items() if k != "embeddings.mask_token"
        }
        save_file(converted, str(out / "model.safetensors"))
        if name == "plain":
            # The untouched PyTorch state dict, to test the converter itself.
            save_file(state, str(out / "pytorch_model.safetensors"))
        print(f"model {name}: {len(converted)} tensors, inputs {list(inputs)}")


def resized(image, size):
    return np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR))


def make_images():
    rng = np.random.default_rng(0)
    out = OUT / "images"
    out.mkdir(parents=True, exist_ok=True)
    expected = {}

    noise = Image.fromarray(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8))
    noise.save(out / "noise.png")
    for size in (32, 64, 37):
        expected[f"noise.{size}"] = resized(noise, size)

    # A small portrait photo: downscaled, upscaled, and the raw values for colour-profile tests.
    photo = Image.open(PHOTO).convert("RGB").resize((99, 148), Image.LANCZOS)
    photo.save(out / "photo.png")
    expected["photo.raw"] = np.asarray(photo)
    for size in (32, 224):
        expected[f"photo.{size}"] = resized(photo, size)

    gray = photo.convert("L")
    gray.save(out / "gray.png")
    expected["gray.32"] = resized(gray, 32)

    # Transparency: the processor composites onto black.
    rgba = np.asarray(photo.crop((0, 0, 64, 64)).convert("RGBA")).copy()
    rgba[..., 3] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    alpha = Image.fromarray(rgba, "RGBA")
    alpha.save(out / "alpha.png")
    black = Image.new("RGBA", alpha.size, (0, 0, 0, 255))
    expected["alpha.over_black.32"] = resized(Image.alpha_composite(black, alpha), 32)

    # The eight EXIF orientations of one small image, turned upright the way Pillow does it.
    stored = Image.fromarray(rng.integers(0, 256, (12, 20, 3), dtype=np.uint8))
    expected["orientation.raw"] = np.asarray(stored)
    upright = {
        1: None, 2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90,
    }
    for orientation, method in upright.items():
        turned = stored if method is None else stored.transpose(method)
        expected[f"orientation.{orientation}.16"] = resized(turned, 16)

    save_numpy(expected, str(out / "expected.safetensors"))
    print(f"images: {sorted(p.name for p in out.iterdir())}, {len(expected)} expected arrays")


if __name__ == "__main__":
    make_models()
    make_images()
