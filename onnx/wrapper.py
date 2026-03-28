import torch
import torch.nn as nn
import torch.nn.functional as F

from unik3d.models.unik3d import get_paddings, get_resize_factor
from unik3d.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD


def coords_grid(b: int, h: int, w: int, device=None) -> torch.Tensor:
    pixel_coords_x = torch.linspace(0.5, w - 0.5, w, device=device)
    pixel_coords_y = torch.linspace(0.5, h - 0.5, h, device=device)
    grid = torch.stack(
        [pixel_coords_x.repeat(h, 1), pixel_coords_y.repeat(w, 1).t()], dim=0
    ).float()
    return grid[None].repeat(b, 1, 1, 1)


def invert_pinhole(k: torch.Tensor) -> torch.Tensor:
    fx = k[..., 0, 0]
    fy = k[..., 1, 1]
    cx = k[..., 0, 2]
    cy = k[..., 1, 2]

    zeros = torch.zeros_like(fx)
    ones = torch.ones_like(fx)

    r0 = torch.stack([1.0 / fx, zeros, -cx / fx], dim=-1)
    r1 = torch.stack([zeros, 1.0 / fy, -cy / fy], dim=-1)
    r2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


class UniK3DOnnxWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_shape,
        resolution_level: int = 9,
        export_features: bool = False,
        feature_layer: int = -1,
    ):
        super().__init__()
        self.model = model
        self.export_features = export_features
        self.feature_layer = feature_layer
        if hasattr(self.model, "pixel_decoder") and hasattr(
            self.model.pixel_decoder, "skip_angular_when_rays_gt"
        ):
            # ONNX path: rays are provided from intrinsics, so skip angular branch compute.
            self.model.pixel_decoder.skip_angular_when_rays_gt = True
        self.input_shape = input_shape
        self.resolution_level = resolution_level

        h, w = input_shape
        ratio_bounds = model.shape_constraints["ratio_bounds"]
        pixels_bounds = [
            model.shape_constraints["pixels_min"],
            model.shape_constraints["pixels_max"],
        ]

        pixels_range = pixels_bounds[1] - pixels_bounds[0]
        interval = pixels_range / 10
        new_lowbound = self.resolution_level * interval + pixels_bounds[0]
        new_upbound = (self.resolution_level + 1) * interval + pixels_bounds[0]
        pixels_bounds = (new_lowbound, new_upbound)

        self.paddings, (padded_h, padded_w) = get_paddings((h, w), ratio_bounds)
        self.resize_factor, (self.new_h, self.new_w) = get_resize_factor(
            (padded_h, padded_w), pixels_bounds
        )

        self.padded_h = padded_h
        self.padded_w = padded_w

        self.register_buffer("mean", torch.tensor(IMAGENET_DATASET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_DATASET_STD).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor):
        # images: [B, 3, H, W] uint8/float
        # intrinsics: [B, 1, 3, 3] or [B, 3, 3] (pinhole only)
        b, _, _, _ = images.shape

        if images.dtype == torch.uint8:
            rgb = images.to(torch.float32) / 255.0
        else:
            rgb = images.to(torch.float32)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0

        rgb = (rgb - self.mean) / self.std

        if intrinsics.dim() == 4:
            k = intrinsics[:, 0, :, :]
        else:
            k = intrinsics

        pad_left, pad_right, pad_top, pad_bottom = self.paddings
        rgb = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        rgb = F.interpolate(
            rgb, size=(self.new_h, self.new_w), mode="bilinear", align_corners=False
        )

        # Scale pinhole intrinsics for pad+resize stage.
        k_scaled = k.clone()
        k_scaled[:, 0, 0] = k_scaled[:, 0, 0] * self.resize_factor
        k_scaled[:, 1, 1] = k_scaled[:, 1, 1] * self.resize_factor
        k_scaled[:, 0, 2] = (k_scaled[:, 0, 2] + pad_left) * self.resize_factor
        k_scaled[:, 1, 2] = (k_scaled[:, 1, 2] + pad_top) * self.resize_factor

        uv = coords_grid(b, self.new_h, self.new_w, device=rgb.device)
        k_inv = invert_pinhole(k_scaled)

        uv_hom = torch.cat(
            [uv, torch.ones(b, 1, self.new_h, self.new_w, device=rgb.device)], dim=1
        )
        uv_flat = uv_hom.reshape(b, 3, -1)
        rays_flat = torch.matmul(k_inv, uv_flat)
        rays = rays_flat.reshape(b, 3, self.new_h, self.new_w)
        rays = rays / torch.norm(rays, dim=1, keepdim=True).clamp(min=1e-4)

        raw_features, raw_tokens = self.model.pixel_encoder(rgb)

        inputs = {
            "image": rgb,
            "rays": rays,
            "features": [
                self.model.stacking_fn(raw_features[i:j]).contiguous()
                for i, j in self.model.slices_encoder_range
            ],
            "tokens": [
                self.model.stacking_fn(raw_tokens[i:j]).contiguous()
                for i, j in self.model.slices_encoder_range
            ],
        }
        model_outputs = self.model.pixel_decoder(inputs, image_metas={})

        rays_out = model_outputs["rays"].permute(0, 2, 1).reshape(
            b, 3, self.new_h, self.new_w
        )
        points_out = rays_out * model_outputs["distance"]

        confidence = F.interpolate(
            model_outputs["confidence"],
            size=(self.padded_h, self.padded_w),
            mode="bilinear",
            align_corners=False,
        )
        confidence = confidence[
            ..., pad_top : self.padded_h - pad_bottom, pad_left : self.padded_w - pad_right
        ]

        points = F.interpolate(
            points_out,
            size=(self.padded_h, self.padded_w),
            mode="bilinear",
            align_corners=False,
        )
        points = points[
            ..., pad_top : self.padded_h - pad_bottom, pad_left : self.padded_w - pad_right
        ]

        depth = points[:, -1:, :, :]
        if not self.export_features:
            return depth, confidence

        num_feature_layers = len(raw_features)
        layer_index = (
            self.feature_layer
            if self.feature_layer >= 0
            else num_feature_layers + self.feature_layer
        )
        if layer_index < 0 or layer_index >= num_feature_layers:
            raise ValueError(
                f"Invalid feature_layer={self.feature_layer}. "
                f"Valid range is [{-num_feature_layers}, {num_feature_layers - 1}]"
            )

        features = raw_features[layer_index].permute(0, 3, 1, 2).contiguous()
        return depth, confidence, features
