#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from arc2face import project_face_embs
from ppid.data import FaceImageDataset
from ppid.models import IdentityPerturbator
from ppid.utils.losses import compute_clip_score, nt_xent_loss, privacy_loss
from ppid.utils.pipeline import (
    build_arc2face_pipeline,
    build_face_analysis,
    extract_arcface_embedding,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_keep(batch):
    images, labels, paths = zip(*batch)
    return list(images), list(labels), list(paths)


def build_label_map(dataset: FaceImageDataset, data_root: str):
    root_name = os.path.basename(data_root.rstrip("/"))
    names = []
    for path in dataset.paths:
        parent = os.path.basename(os.path.dirname(path))
        names.append(parent if parent != root_name else os.path.splitext(os.path.basename(path))[0])
    return {n: i for i, n in enumerate(sorted(set(names)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--arc2face_model_dir", type=str, default="./models")
    p.add_argument("--insightface_root", type=str, default="./")
    p.add_argument("--sd_model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout_p", type=float, default=0.1)
    p.add_argument("--max_images", type=int, default=8000)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--lambda_quality", type=float, default=0.5)
    p.add_argument("--lambda_contrastive", type=float, default=0.1)
    p.add_argument("--lambda_ce", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_generation", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = FaceImageDataset(args.data_root, max_images=args.max_images)
    name_to_id = build_label_map(dataset, args.data_root)
    num_classes = max(len(name_to_id), 2)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_keep
    )

    face_app = build_face_analysis(args.insightface_root)
    pipeline = None
    if not args.skip_generation:
        pipeline = build_arc2face_pipeline(
            args.arc2face_model_dir, sd_model=args.sd_model, device=str(device)
        )
        pipeline.unet.eval()
        pipeline.text_encoder.eval()
        for param in list(pipeline.unet.parameters()) + list(pipeline.text_encoder.parameters()):
            param.requires_grad = False

    perturbor = IdentityPerturbator(dim=512, dropout_p=args.dropout_p).to(device)
    classifier = nn.Linear(512, num_classes).to(device)
    optimizer = torch.optim.Adam(
        list(perturbor.parameters()) + list(classifier.parameters()), lr=args.lr
    )

    best_loss = float("inf")
    for epoch in range(args.epochs):
        perturbor.train()
        classifier.train()
        running = defaultdict(float)
        n_steps = 0

        for images, labels, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            embs, y = [], []
            for img, lab in zip(images, labels):
                try:
                    embs.append(extract_arcface_embedding(face_app, img, device=device))
                    y.append(name_to_id.get(lab, 0))
                except ValueError:
                    continue
            if len(embs) < 2:
                continue

            id_emb = torch.stack(embs, dim=0)
            y = torch.tensor(y[: id_emb.size(0)], device=device, dtype=torch.long)
            z1 = perturbor(id_emb)
            z2 = perturbor(id_emb)

            loss_contrastive = nt_xent_loss(z1, z2, temperature=0.1)
            loss_ce = F.cross_entropy(classifier(z1), y)
            loss_priv_emb = privacy_loss(id_emb.detach(), z1)

            loss_qual = torch.tensor(0.0, device=device)
            loss_priv_img = torch.tensor(0.0, device=device)
            if pipeline is not None:
                with torch.no_grad():
                    prompt = project_face_embs(pipeline, z1.detach().half())
                    gen_imgs = pipeline(
                        prompt_embeds=prompt,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=1,
                    ).images
                    anon_embs = []
                    for g in gen_imgs:
                        try:
                            anon_embs.append(extract_arcface_embedding(face_app, g, device=device))
                        except ValueError:
                            anon_embs.append(torch.zeros(512, device=device))
                    loss_priv_img = privacy_loss(id_emb.detach(), torch.stack(anon_embs, dim=0))
                    try:
                        loss_qual = (1.0 - compute_clip_score(gen_imgs, device=str(device)).float()).mean()
                    except Exception:
                        pass

            train_loss = (
                loss_priv_emb
                + args.lambda_contrastive * loss_contrastive
                + args.lambda_ce * loss_ce
            )
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running["loss"] += float(
                (loss_priv_img if pipeline else loss_priv_emb).detach()
                + args.lambda_quality * loss_qual.detach()
                + args.lambda_contrastive * loss_contrastive.detach()
                + args.lambda_ce * loss_ce.detach()
            )
            running["priv"] += float((loss_priv_img if pipeline else loss_priv_emb).detach())
            running["qual"] += float(loss_qual.detach())
            running["con"] += float(loss_contrastive.detach())
            running["ce"] += float(loss_ce.detach())
            n_steps += 1

        if n_steps == 0:
            continue

        avg = {k: v / n_steps for k, v in running.items()}
        print(
            f"[Epoch {epoch}] loss={avg['loss']:.4f} priv={avg['priv']:.4f} "
            f"qual={avg['qual']:.4f} con={avg['con']:.4f} ce={avg['ce']:.4f}"
        )
        torch.save(
            {"epoch": epoch, "perturbor": perturbor.state_dict(), "classifier": classifier.state_dict()},
            os.path.join(args.output_dir, f"perturbor_epoch{epoch}.pth"),
        )
        if avg["loss"] < best_loss:
            best_loss = avg["loss"]
            torch.save(perturbor.state_dict(), os.path.join(args.output_dir, "perturbor_best.pth"))


if __name__ == "__main__":
    main()
