import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Add parent directory to path to import unik3d and wrapper
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(current_dir))

from unik3d.models import UniK3D
from pca_utils import pca_rgb
from wrapper import UniK3DOnnxWrapper


def _to_vis_u8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    vals = x[finite]
    lo = np.percentile(vals, 2.0)
    hi = np.percentile(vals, 98.0)
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _load_demo_image_uint8(image_path: Path, target_h: int, target_w: int) -> np.ndarray:
    img = Image.open(image_path).convert("RGB").resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)  # HWC
    return np.transpose(arr, (2, 0, 1))[None, ...]  # 1CHW


def run_onnx_demo(
    onnx_path: Path,
    image_path: Path,
    input_h: int,
    input_w: int,
    batch_size: int = 1,
    depth_out_path: Path | None = None,
) -> None:
    import onnxruntime as ort

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Demo image not found: {image_path}")

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]

    images = _load_demo_image_uint8(image_path, input_h, input_w)
    if batch_size > 1:
        varied = [images]
        for i in range(1, batch_size):
            varied.append(np.flip(images, axis=3) if i % 2 == 1 else np.flip(images, axis=2))
        images = np.concatenate(varied, axis=0)

    actual_batch = images.shape[0]
    intrinsics = np.eye(3, dtype=np.float32).reshape(1, 1, 3, 3).repeat(actual_batch, axis=0)
    intrinsics[:, 0, 0, 0] = float(input_w)
    intrinsics[:, 0, 1, 1] = float(input_h)
    intrinsics[:, 0, 0, 2] = float(input_w) / 2.0
    intrinsics[:, 0, 1, 2] = float(input_h) / 2.0

    inputs = {}
    if "images" in input_names:
        inputs["images"] = images
    if "intrinsics" in input_names:
        inputs["intrinsics"] = intrinsics

    outputs = sess.run(output_names, inputs)
    out_dict = dict(zip(output_names, outputs))

    print("[DEMO] Inference complete. Output shapes:")
    for name, val in out_dict.items():
        print(f"  - {name}: {val.shape} ({val.dtype})")

    depth_val = out_dict["depth"]
    conf_val = out_dict["confidence"] if "confidence" in out_dict else None
    features_val = out_dict["features"] if "features" in out_dict else None

    depth0 = depth_val[0, 0].astype(np.float32)
    print(
        f"[DEMO] Depth stats (batch 0): "
        f"min={depth0.min():.4f}, max={depth0.max():.4f}, mean={depth0.mean():.4f}"
    )

    if conf_val is not None:
        conf0 = conf_val[0, 0].astype(np.float32)
        print(
            f"[DEMO] Confidence(raw uncertainty) stats (batch 0): "
            f"min={conf0.min():.4f}, max={conf0.max():.4f}, mean={conf0.mean():.4f}"
        )
    if features_val is not None:
        feat0 = features_val[0].astype(np.float32)
        print(
            f"[DEMO] Features stats (batch 0): "
            f"min={feat0.min():.4f}, max={feat0.max():.4f}, mean={feat0.mean():.4f}"
        )

    if depth_out_path is None:
        base_depth = image_path.with_name(f"{image_path.stem}_depth.png")
    else:
        base_depth = depth_out_path
    base_conf = base_depth.with_name(f"{base_depth.stem}_confidence{base_depth.suffix}")
    base_feat = base_depth.with_name(f"{base_depth.stem}_features_pca{base_depth.suffix}")

    for i in range(actual_batch):
        d_i = depth_val[i, 0].astype(np.float32)
        d_vis = _to_vis_u8(d_i)
        if actual_batch > 1:
            depth_path = base_depth.with_name(f"{base_depth.stem}_batch{i}{base_depth.suffix}")
            conf_path = base_conf.with_name(f"{base_conf.stem}_batch{i}{base_conf.suffix}")
            feat_path = base_feat.with_name(f"{base_feat.stem}_batch{i}{base_feat.suffix}")
        else:
            depth_path = base_depth
            conf_path = base_conf
            feat_path = base_feat

        Image.fromarray(d_vis).save(depth_path)
        print(f"[DEMO] Saved depth visualization: {depth_path}")

        if conf_val is not None:
            c_i = conf_val[i, 0].astype(np.float32)
            c_vis = _to_vis_u8(c_i)
            Image.fromarray(c_vis).save(conf_path)
            print(f"[DEMO] Saved confidence visualization: {conf_path}")

        if features_val is not None:
            feat_i = features_val[i]
            feat_vis = pca_rgb(feat_i, input_layout="CHW")
            feat_img = Image.fromarray(feat_vis, mode="RGB").resize(
                (depth_val.shape[-1], depth_val.shape[-2]), Image.BILINEAR
            )
            feat_img.save(feat_path)
            print(f"[DEMO] Saved features PCA visualization: {feat_path}")


def export_onnx(args):
    print(f"Loading UniK3D model: unik3d-{args.model_size}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{args.model_size}")
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    model = model.to(device).eval()

    input_shape = (args.input_height, args.input_width)
    wrapper = UniK3DOnnxWrapper(
        model,
        input_shape,
        resolution_level=args.resolution_level,
        export_features=args.export_features,
        feature_layer=args.feature_layer,
    )
    wrapper = wrapper.to(device).eval()

    batch_size = args.fixed_batch_size

    # DA3-style image input: uint8 NCHW
    dummy_images = torch.randint(
        0,
        256,
        (batch_size, 3, args.input_height, args.input_width),
        dtype=torch.uint8,
        device=device,
    )

    # Pinhole intrinsics: [B, 1, 3, 3]
    dummy_intrinsics = torch.eye(3, dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1)
    dummy_intrinsics[:, 0, 0, 0] = float(args.input_width)
    dummy_intrinsics[:, 0, 1, 1] = float(args.input_height)
    dummy_intrinsics[:, 0, 0, 2] = float(args.input_width) / 2.0
    dummy_intrinsics[:, 0, 1, 2] = float(args.input_height) / 2.0

    output_path = args.output_path
    if not output_path:
        output_path = os.path.join(current_dir, f"unik3d_{args.model_size}.onnx")

    print(
        f"Exporting to {output_path} with fixed batch size {batch_size}..."
    )

    input_names = ["images", "intrinsics"]
    output_names = ["depth", "confidence"]
    if args.export_features:
        output_names.append("features")

    torch.onnx.export(
        wrapper,
        (dummy_images, dummy_intrinsics),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
        do_constant_folding=False,
        export_params=True,
        verbose=False,
        external_data=False,
    )

    print("Export complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export UniK3D to ONNX")
    parser.add_argument("--model_size", type=str, default="vitb", choices=["vits", "vitb", "vitl"], help="Model size")
    parser.add_argument("--resolution_level", type=int, default=9, help="Resolution level (0-9)")
    parser.add_argument("--input_height", type=int, default=336, help="Input image height")
    parser.add_argument("--input_width", type=int, default=448, help="Input image width")
    parser.add_argument("--output_path", type=str, default="", help="Output ONNX file path")
    parser.add_argument("--fixed_batch_size", type=int, default=1, help="Fixed batch size for export.")
    parser.add_argument(
        "--export_features",
        action="store_true",
        help="If set, export an additional 'features' output from an intermediate encoder layer.",
    )
    parser.add_argument(
        "--feature_layer",
        type=int,
        default=-1,
        help="Raw encoder block index for 'features' output (supports negative indexing).",
    )
    parser.add_argument(
        "--demo-image",
        type=str,
        default=str(current_dir / "examples" / "image.jpg"),
        help="Path to demo image for ONNX inference after export.",
    )
    parser.add_argument(
        "--demo-output",
        type=str,
        default="",
        help="Optional output path for depth visualization PNG (confidence uses sibling filename).",
    )

    args = parser.parse_args()
    export_onnx(args)
    if args.demo_image:
        try:
            run_onnx_demo(
                onnx_path=Path(args.output_path) if args.output_path else current_dir / f"unik3d_{args.model_size}.onnx",
                image_path=Path(args.demo_image),
                input_h=args.input_height,
                input_w=args.input_width,
                batch_size=args.fixed_batch_size,
                depth_out_path=Path(args.demo_output) if args.demo_output else None,
            )
        except ImportError as exc:
            print(f"[DEMO] Skipping demo (missing dependency): {exc}")
        except Exception as exc:
            print(f"[DEMO] Demo failed: {exc}")
