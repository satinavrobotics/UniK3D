#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# Add parent directory to path to import unik3d
CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
sys.path.append(str(PARENT_DIR))

from unik3d.models import UniK3D
from unik3d.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD
from pca_utils import pca_rgb


def parse_layers(layers_str: str) -> list[int]:
    parts = [x.strip() for x in layers_str.split(",") if x.strip()]
    if not parts:
        raise ValueError("No layers provided. Example: --layers 3,6,9,-1")
    return [int(x) for x in parts]


def load_and_preprocess_image(
    image_path: Path, input_h: int, input_w: int, device: torch.device
) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((input_w, input_h), Image.BILINEAR)
    image_np = np.asarray(image_resized, dtype=np.float32) / 255.0  # HWC
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # 1CHW

    mean = torch.tensor(IMAGENET_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DATASET_STD, device=device).view(1, 3, 1, 1)
    image_t = (image_t - mean) / std
    return image_t, image_resized


def normalize_layer_index(layer: int, n_layers: int) -> int:
    idx = layer if layer >= 0 else n_layers + layer
    if idx < 0 or idx >= n_layers:
        raise ValueError(
            f"Invalid layer {layer}. Valid range is [{-n_layers}, {n_layers - 1}]"
        )
    return idx


def save_grid(
    images: list[Image.Image],
    labels: list[str],
    output_path: Path,
    cols: int = 3,
    pad: int = 12,
    label_h: int = 26,
) -> None:
    if not images:
        return
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)
    canvas_w = cols * w + (cols + 1) * pad
    canvas_h = rows * (h + label_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(images, labels)):
        r = i // cols
        c = i % cols
        x0 = pad + c * (w + pad)
        y0 = pad + r * (h + label_h + pad)
        canvas.paste(img, (x0, y0))
        draw.text((x0, y0 + h + 4), label, fill=(230, 230, 230))

    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize UniK3D encoder patch features via PCA."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=str(CURRENT_DIR / "examples" / "image.jpg"),
        help="Path to input RGB image.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="vitb",
        choices=["vits", "vitb", "vitl"],
        help="UniK3D model size.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="3,6,9,-1",
        help="Comma-separated raw encoder block indices. Supports negative indexing.",
    )
    parser.add_argument(
        "--input_height",
        type=int,
        default=336,
        help="Resize height before feature extraction.",
    )
    parser.add_argument(
        "--input_width",
        type=int,
        default=448,
        help="Resize width before feature extraction.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(CURRENT_DIR / "examples" / "pca_layers"),
        help="Directory for output images.",
    )
    parser.add_argument(
        "--resolution_level",
        type=int,
        default=9,
        help="Model resolution level (for consistency with exporter setup).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading model unik3d-{args.model_size} on {device} ...")
    model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{args.model_size}")
    model.resolution_level = args.resolution_level
    model = model.to(device).eval()

    requested_layers = parse_layers(args.layers)
    with torch.no_grad():
        image_t, image_resized = load_and_preprocess_image(
            image_path, args.input_height, args.input_width, device
        )
        raw_features, _ = model.pixel_encoder(image_t)

    n_layers = len(raw_features)
    print(f"Encoder produced {n_layers} raw feature maps.")

    # Save input preview.
    input_out = output_dir / "input_resized.png"
    image_resized.save(input_out)
    print(f"Saved input preview: {input_out}")

    grid_images: list[Image.Image] = [image_resized.copy()]
    grid_labels: list[str] = ["input"]

    for layer in requested_layers:
        idx = normalize_layer_index(layer, n_layers)
        rgb_u8 = pca_rgb(raw_features[idx].detach().cpu().numpy(), input_layout="HWC")
        pca_img = Image.fromarray(rgb_u8, mode="RGB").resize(
            (args.input_width, args.input_height), Image.BILINEAR
        )
        out_path = output_dir / f"layer_{idx:02d}_pca.png"
        pca_img.save(out_path)

        h_f, w_f = raw_features[idx].shape[1], raw_features[idx].shape[2]
        c_f = raw_features[idx].shape[3]
        print(
            f"Layer {layer} -> idx {idx}: feature shape [1,{h_f},{w_f},{c_f}] saved to {out_path}"
        )

        grid_images.append(pca_img)
        grid_labels.append(f"layer {layer} -> {idx}")

    grid_path = output_dir / "pca_grid.png"
    save_grid(grid_images, grid_labels, grid_path, cols=3)
    print(f"Saved summary grid: {grid_path}")


if __name__ == "__main__":
    main()
