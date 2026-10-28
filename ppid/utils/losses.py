from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import clip
except Exception:
    clip = None

_clip_model = None
_clip_preprocess = None


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1)
    b = F.normalize(b.float(), dim=-1)
    return (a * b).sum(dim=-1)


def privacy_loss(orig_emb: torch.Tensor, anon_emb: torch.Tensor) -> torch.Tensor:
    return cosine_sim(orig_emb, anon_emb).mean()


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    z_i = F.normalize(z_i.float(), dim=-1)
    z_j = F.normalize(z_j.float(), dim=-1)
    batch = z_i.size(0)
    rep = torch.cat([z_i, z_j], dim=0)
    sim = rep @ rep.T / temperature
    sim = sim.masked_fill(torch.eye(2 * batch, device=sim.device, dtype=torch.bool), float("-inf"))
    positives = torch.cat(
        [torch.arange(batch, 2 * batch, device=sim.device), torch.arange(0, batch, device=sim.device)],
        dim=0,
    )
    return F.cross_entropy(sim, positives)


@torch.no_grad()
def compute_clip_score(images, prompt: str = "a photo of a human face", device: str = "cuda") -> torch.Tensor:
    global _clip_model, _clip_preprocess
    if clip is None:
        raise ImportError("pip install git+https://github.com/openai/CLIP.git")
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    if not isinstance(images, (list, tuple)):
        images = [images]
    image_inputs = torch.stack([_clip_preprocess(img) for img in images]).to(device)
    text_inputs = clip.tokenize([prompt], device=device)
    image_features = F.normalize(_clip_model.encode_image(image_inputs).float(), dim=-1)
    text_features = F.normalize(_clip_model.encode_text(text_inputs).float(), dim=-1)
    return (image_features @ text_features.T).squeeze(-1)
