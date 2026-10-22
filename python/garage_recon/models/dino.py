"""DINOv2-based geometric correspondence (README 8.4).

* RomaMatcher (Matcher): RoMa (Edstedt et al., CVPR 2024; MIT) - DINOv2
  ViT-L/14 coarse matching refined by convolutional features to a dense,
  sub-pixel warp with a certainty. With query points it samples the warp
  there, so matches of one anchor image chain into multi-view tracks (the
  camera calibration stage).
* Dinov2Features (DenseFeatures): DINOv2 ViT-L/14 patch tokens (stride 14).
* vlad_descriptors: AnyLoc-style (Keetha et al., RA-L 2023) global image
  descriptors, VLAD over the patch tokens with k-means words learned on the
  images themselves, for appearance loop / multi-trip candidates. They rank
  candidates; ICP + PCM still decide.

Weights: RoMa outdoor and DINOv2 ViT-L/14 (Apache 2.0), from a local
directory (models.dino.weights) or their release URLs.
"""

from __future__ import annotations

import pathlib

import numpy as np

from garage_recon.models import interfaces

_DINOV2 = "dinov2_vitl14_pretrain.pth"
_ROMA = "roma_outdoor.pth"


def _weights(directory, name, url):
    import torch
    path = pathlib.Path(directory) / name if directory else None
    if path is not None and path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return torch.hub.load_state_dict_from_url(url, map_location="cpu")


def _dinov2_state(directory):
    return _weights(directory, _DINOV2, "https://dl.fbaipublicfiles.com/"
                    "dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth")


class RomaMatcher:
    """Dense matcher on RoMa (outdoor weights)."""

    needs_images = True

    def __init__(self, weights: str | None = None, device: str = "cuda",
                 coarse_res: int = 560, upsample_res: int = 864,
                 max_matches: int = 2000, min_certainty: float = 0.3,
                 grid_step: int = 16):
        import torch
        from romatch import roma_outdoor
        self.torch = torch
        self.device = device
        self.model = roma_outdoor(
            device=device,
            weights=_weights(weights, _ROMA, "https://github.com/Parskatt/"
                             "storage/releases/download/roma/roma_outdoor.pth"),
            dinov2_weights=_dinov2_state(weights), coarse_res=coarse_res,
            upsample_res=upsample_res,
            use_custom_corr=False)  # Plain PyTorch local correlation.
        self.max_matches = int(max_matches)
        self.min_certainty = float(min_certainty)
        self.grid_step = int(grid_step)
        self.name = "roma_outdoor"

    def query_points(self, frame: interfaces.Frame) -> np.ndarray:
        """A regular grid of query pixels."""
        h, w = frame.image.shape[:2]
        s = self.grid_step
        u, v = np.meshgrid(np.arange(s // 2, w, s), np.arange(s // 2, h, s))
        return np.column_stack([u.ravel(), v.ravel()]).astype(np.float64)

    def match(self, frame_a: interfaces.Frame, frame_b: interfaces.Frame,
              points_a: np.ndarray | None = None) -> interfaces.Matches:
        from PIL import Image
        torch = self.torch
        image_a = Image.fromarray(frame_a.image)
        image_b = Image.fromarray(frame_b.image)
        ha, wa = frame_a.image.shape[:2]
        hb, wb = frame_b.image.shape[:2]
        with torch.inference_mode():
            warp, certainty = self.model.match(image_a, image_b,
                                               device=self.device)
            if points_a is None:
                matches, score = self.model.sample(warp, certainty,
                                                   num=self.max_matches)
                ka, kb = self.model.to_pixel_coordinates(matches, ha, wa, hb,
                                                         wb)
                score = score.float().cpu().numpy()
                keep = score >= self.min_certainty
                return interfaces.Matches(ka.cpu().numpy()[keep],
                                          kb.cpu().numpy()[keep], score[keep])
            # The A -> B half of the (symmetric) warp, sampled at the queries.
            if warp.dim() == 4:  # Batched output.
                warp, certainty = warp[0], certainty[0]
            w = warp.shape[1] // 2 if self.model.symmetric else warp.shape[1]
            grid_ab = warp[:, :w, 2:].permute(2, 0, 1)[None].float()
            cert_ab = certainty[:, :w][None, None].float()
            q = torch.tensor(points_a, device=self.device, dtype=torch.float32)
            norm = torch.stack([(q[:, 0] + 0.5) / wa * 2 - 1,
                                (q[:, 1] + 0.5) / ha * 2 - 1], dim=-1)
            sample = norm[None, None]
            target = torch.nn.functional.grid_sample(
                grid_ab, sample, align_corners=False)[0, :, 0].T
            score = torch.nn.functional.grid_sample(
                cert_ab, sample, align_corners=False)[0, 0, 0]
            uv_b = torch.stack([(target[:, 0] + 1) / 2 * wb - 0.5,
                                (target[:, 1] + 1) / 2 * hb - 0.5], dim=-1)
            score = score.cpu().numpy()
            uv_b = uv_b.cpu().numpy()
        inside = ((uv_b[:, 0] >= 0) & (uv_b[:, 0] < wb) & (uv_b[:, 1] >= 0)
                  & (uv_b[:, 1] < hb))
        keep = inside & (score >= self.min_certainty)
        index = np.nonzero(keep)[0]
        return interfaces.Matches(points_a[keep], uv_b[keep], score[keep],
                                  index)


class Dinov2Features:
    """DINOv2 ViT-L/14 patch tokens; with `ade20k_head` (Meta's linear
    ADE20K head on the last-layer tokens, Apache 2.0) also semantic classes,
    e.g. the sky."""

    needs_images = True
    ADE20K_SKY = 2  # Class index in ADE20K-150.

    def __init__(self, weights: str | None = None, device: str = "cuda",
                 resolution: int = 518, ade20k_head: str | None = None):
        import torch
        from romatch.models.transformer import vit_large
        self.torch = torch
        self.device = device
        model = vit_large(patch_size=14, img_size=518, init_values=1.0,
                          ffn_layer="mlp", block_chunks=0)
        model.load_state_dict(_dinov2_state(weights))
        self.model = model.to(device).eval().half()
        self.resolution = int(resolution)
        self.stride = 14
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
        self.head = None
        if ade20k_head:
            sd = torch.load(ade20k_head, map_location=device,
                            weights_only=False)["state_dict"]
            bn = {k.rsplit(".", 1)[1]: v.float() for k, v in sd.items()
                  if k.startswith("decode_head.bn.")}
            scale = bn["weight"] / torch.sqrt(bn["running_var"] + 1e-5)
            # BatchNorm folded into the 1x1 convolution.
            weight = sd["decode_head.conv_seg.weight"][:, :, 0, 0].float()
            self.head = (weight * scale[None], sd["decode_head.conv_seg.bias"].float()
                         + weight @ (bn["bias"] - bn["running_mean"] * scale))

    def _tokens(self, image: np.ndarray):
        torch = self.torch
        h, w = image.shape[:2]
        scale = self.resolution / max(h, w)
        th = max(14, int(round(h * scale / 14)) * 14)
        tw = max(14, int(round(w * scale / 14)) * 14)
        x = torch.tensor(image, device=self.device).permute(2, 0, 1).float() / 255
        x = torch.nn.functional.interpolate(x[None], (th, tw), mode="bilinear",
                                            align_corners=False)[0]
        x = ((x - self.mean) / self.std)[None].half()
        with torch.inference_mode():
            out = self.model.forward_features(x)
        tokens = out["x_norm_patchtokens"][0].float()
        return tokens.reshape(th // 14, tw // 14, -1), h / th * 14

    def features(self, frame: interfaces.Frame) -> tuple[np.ndarray, int]:
        tokens, stride = self._tokens(frame.image)
        return tokens.cpu().numpy(), int(round(stride))

    def tokens(self, image: np.ndarray):
        """(h, w, D) torch patch tokens of an RGB image (GPU)."""
        return self._tokens(image)[0]

    def class_probability(self, image: np.ndarray, label: int) -> np.ndarray:
        """(H, W) float32 probability of ADE20K class `label`, at the image
        size (patch logits bilinearly upsampled). Needs `ade20k_head`."""
        if self.head is None:
            raise ValueError("Dinov2Features: no ade20k_head configured")
        torch = self.torch
        weight, bias = self.head
        logits = self._tokens(image)[0] @ weight.T + bias
        p = torch.softmax(logits, dim=-1)[..., label]
        h, w = image.shape[:2]
        return torch.nn.functional.interpolate(
            p[None, None], (h, w), mode="bilinear",
            align_corners=False)[0, 0].cpu().numpy().astype(np.float32)


def vlad_descriptors(token_sets, num_words: int = 32, iterations: int = 20,
                     sample: int = 200000, seed: int = 0) -> np.ndarray:
    """AnyLoc-style VLAD: k-means words on the tokens, residuals summed per
    word, intra- and L2-normalised. `token_sets`: list of (h, w, D) torch
    tensors on one device. Returns (N, num_words * D) float32."""
    import torch
    device = token_sets[0].device
    flat = [t.reshape(-1, t.shape[-1]) for t in token_sets]
    flat = [torch.nn.functional.normalize(f, dim=1) for f in flat]
    pool = torch.cat(flat)
    g = torch.Generator(device="cpu").manual_seed(seed)
    pick = torch.randperm(len(pool), generator=g)[:sample].to(device)
    data = pool[pick]
    words = data[torch.randperm(len(data), generator=g)[:num_words].to(device)]
    for _ in range(iterations):  # Spherical k-means.
        assign = (data @ words.T).argmax(dim=1)
        for k in range(num_words):
            members = data[assign == k]
            if len(members):
                words[k] = torch.nn.functional.normalize(members.mean(0), dim=0)
    out = []
    for f in flat:
        assign = (f @ words.T).argmax(dim=1)
        v = torch.zeros_like(words)
        v.index_add_(0, assign, f - words[assign])
        v = torch.nn.functional.normalize(v, dim=1)  # Intra-normalisation.
        out.append(torch.nn.functional.normalize(v.reshape(-1), dim=0))
    return torch.stack(out).cpu().numpy().astype(np.float32)
