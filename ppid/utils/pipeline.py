from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image


def build_arc2face_pipeline(
    arc2face_model_dir: str,
    sd_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    device: str = "cuda",
    dtype=torch.float16,
):
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from arc2face import CLIPTextModelWrapper

    encoder = CLIPTextModelWrapper.from_pretrained(
        arc2face_model_dir, subfolder="encoder", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        arc2face_model_dir, subfolder="arc2face", torch_dtype=dtype
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        sd_model,
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return pipeline.to(device)


def build_face_analysis(insightface_root: str = "./", det_size=(160, 160), ctx_id: int = 0):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="antelopev2",
        root=insightface_root,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app


def extract_arcface_embedding(
    app,
    image,
    device: Optional[torch.device] = None,
    dtype=torch.float32,
) -> torch.Tensor:
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        arr = np.array(image)[:, :, ::-1]
    else:
        arr = image[:, :, ::-1] if image.shape[-1] == 3 else image

    faces = app.get(arr)
    if not faces:
        raise ValueError("No face detected.")
    face = sorted(
        faces,
        key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
    )[-1]
    emb = torch.tensor(face["embedding"], dtype=dtype)
    emb = emb / emb.norm()
    if device is not None:
        emb = emb.to(device)
    return emb
