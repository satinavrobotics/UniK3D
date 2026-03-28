import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

# Add parent directory to path to import unik3d and wrapper
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
sys.path.append(str(current_dir))

from unik3d.models import UniK3D
from wrapper import UniK3DOnnxWrapper


def verify_onnx(args):
    print(f"Verifying ONNX model: {args.model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading PyTorch model: unik3d-{args.model_size}")
    pt_model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{args.model_size}")
    pt_model.resolution_level = args.resolution_level
    pt_model.interpolation_mode = "bilinear"
    pt_model = pt_model.to(device).eval()

    input_shape = (args.input_height, args.input_width)
    wrapper = UniK3DOnnxWrapper(
        pt_model,
        input_shape,
        resolution_level=args.resolution_level,
        export_features=args.export_features,
        feature_layer=args.feature_layer,
    )
    wrapper = wrapper.to(device).eval()

    print("Loading ONNX model...")
    ort_session = ort.InferenceSession(args.model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    batch_size = args.fixed_batch_size

    dummy_images = torch.randint(
        0,
        256,
        (batch_size, 3, args.input_height, args.input_width),
        dtype=torch.uint8,
        device=device,
    )

    dummy_intrinsics = torch.eye(3, dtype=torch.float32, device=device).view(1, 1, 3, 3).repeat(batch_size, 1, 1, 1)
    dummy_intrinsics[:, 0, 0, 0] = float(args.input_width)
    dummy_intrinsics[:, 0, 1, 1] = float(args.input_height)
    dummy_intrinsics[:, 0, 0, 2] = float(args.input_width) / 2.0
    dummy_intrinsics[:, 0, 1, 2] = float(args.input_height) / 2.0

    print("Running PyTorch inference...")
    with torch.no_grad():
        pt_outputs = wrapper(dummy_images, dummy_intrinsics)
    pt_depth = pt_outputs[0]
    pt_confidence = pt_outputs[1]
    pt_features = pt_outputs[2] if len(pt_outputs) > 2 else None

    print("Running ONNX inference...")
    onnx_inputs = {
        "images": dummy_images.cpu().numpy(),
        "intrinsics": dummy_intrinsics.cpu().numpy(),
    }
    onnx_output_names = [output.name for output in ort_session.get_outputs()]
    onnx_outputs = ort_session.run(onnx_output_names, onnx_inputs)
    onnx_out = dict(zip(onnx_output_names, onnx_outputs))
    onnx_depth = onnx_out["depth"]
    onnx_confidence = onnx_out["confidence"]
    onnx_features = onnx_out.get("features")

    print("Comparing results...")

    def compare(name, pt_out, onnx_out, rtol=1e-3, atol=1e-4):
        pt_np = pt_out.cpu().numpy()
        try:
            np.testing.assert_allclose(pt_np, onnx_out, rtol=rtol, atol=atol)
            print(f"✅ {name} matches!")
            return True
        except AssertionError as e:
            print(f"❌ {name} mismatch!")
            print(e)
            return False

    success = True
    success &= compare("Depth", pt_depth, onnx_depth)
    success &= compare("Confidence (raw uncertainty)", pt_confidence, onnx_confidence)
    if pt_features is not None or onnx_features is not None:
        if pt_features is None or onnx_features is None:
            print("❌ Features mismatch! One output is missing.")
            success = False
        else:
            success &= compare("Features", pt_features, onnx_features)

    if success:
        print("\nVerification SUCCESSFUL! The ONNX model matches the PyTorch model.")
    else:
        print("\nVerification FAILED! Mismatches detected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify UniK3D ONNX model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--model_size", type=str, default="vits", choices=["vits", "vitb", "vitl"], help="Model size")
    parser.add_argument("--resolution_level", type=int, default=9, help="Resolution level (0-9)")
    parser.add_argument("--input_height", type=int, default=240, help="Input image height")
    parser.add_argument("--input_width", type=int, default=320, help="Input image width")
    parser.add_argument("--fixed_batch_size", type=int, default=2, help="Verify with fixed batch size.")
    parser.add_argument(
        "--export_features",
        action="store_true",
        help="Expect and verify an additional 'features' output.",
    )
    parser.add_argument(
        "--feature_layer",
        type=int,
        default=-1,
        help="Raw encoder block index for 'features' output (supports negative indexing).",
    )

    args = parser.parse_args()
    verify_onnx(args)
