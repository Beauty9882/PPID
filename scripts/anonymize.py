#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import torch
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from arc2face import project_face_embs
from ppid.models import IdentityPerturbator
from ppid.utils.pipeline import build_arc2face_pipeline, build_face_analysis, extract_arcface_embedding

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root: str):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in IMG_EXTS:
                files.append(os.path.join(dirpath, name))
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True)
    p.add_argument("--perturbor_ckpt", type=str, default="./checkpoints/perturbor_best.pth")
    p.add_argument("--arc2face_model_dir", type=str, default="./models")
    p.add_argument("--insightface_root", type=str, default="./")
    p.add_argument("--sd_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="./outputs/anonymized")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_images", type=int, default=1)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    face_app = build_face_analysis(args.insightface_root)
    pipeline = build_arc2face_pipeline(
        args.arc2face_model_dir, sd_model=args.sd_model, device=str(device)
    )

    perturbor = IdentityPerturbator(dim=512, dropout_p=args.dropout_p).to(device)
    state = torch.load(args.perturbor_ckpt, map_location=device)
    if isinstance(state, dict) and "perturbor" in state:
        state = state["perturbor"]
    perturbor.load_state_dict(state)
    perturbor.eval()

    generator = torch.Generator(device=device).manual_seed(args.seed)
    paths = list_images(args.image_dir)

    for path in tqdm(paths):
        try:
            id_emb = extract_arcface_embedding(face_app, path, device=device)
        except ValueError:
            continue

        with torch.no_grad():
            z_tilde = perturbor(id_emb.unsqueeze(0).float()).half()
            prompt = project_face_embs(pipeline, z_tilde)
            images = pipeline(
                prompt_embeds=prompt,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=args.num_images,
                generator=generator,
            ).images

        rel = os.path.relpath(path, args.image_dir)
        out_base = os.path.join(args.output_dir, os.path.splitext(rel)[0])
        os.makedirs(os.path.dirname(out_base) or args.output_dir, exist_ok=True)
        for i, img in enumerate(images):
            suffix = "" if args.num_images == 1 else f"_{i}"
            img.save(f"{out_base}{suffix}.png")


if __name__ == "__main__":
    main()
