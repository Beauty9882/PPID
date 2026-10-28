#!/usr/bin/env python3
import argparse
import os

import lpips
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder_a", type=str, required=True)
    p.add_argument("--folder_b", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--size", type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    to_tensor = T.Compose([T.Resize((args.size, args.size)), T.ToTensor()])
    lpips_model = lpips.LPIPS(net="vgg").to(device).eval()

    names = sorted(
        f for f in os.listdir(args.folder_a)
        if os.path.splitext(f)[1].lower() in {".png", ".jpg", ".jpeg"}
    )
    ssim_scores, lpips_scores = [], []
    for name in names:
        path_b = os.path.join(args.folder_b, name)
        if not os.path.exists(path_b):
            alt = os.path.join(args.folder_b, os.path.splitext(name)[0] + ".png")
            if not os.path.exists(alt):
                continue
            path_b = alt

        img1 = Image.open(os.path.join(args.folder_a, name)).convert("RGB")
        img2 = Image.open(path_b).convert("RGB")
        with torch.no_grad():
            lpips_scores.append(
                lpips_model(
                    to_tensor(img1).unsqueeze(0).to(device),
                    to_tensor(img2).unsqueeze(0).to(device),
                ).item()
            )
        a = np.asarray(img1.resize((args.size, args.size)), dtype=np.float32) / 255.0
        b = np.asarray(img2.resize((args.size, args.size)), dtype=np.float32) / 255.0
        ssim_scores.append(ssim(a, b, channel_axis=2, data_range=1.0))

    print(f"Pairs: {len(ssim_scores)}")
    print(f"Average SSIM:  {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"Average LPIPS: {sum(lpips_scores)/len(lpips_scores):.4f}")


if __name__ == "__main__":
    main()
