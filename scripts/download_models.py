#!/usr/bin/env python3
import argparse
import os

from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", type=str, default="./models")
    args = p.parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    for filename in [
        "arc2face/config.json",
        "arc2face/diffusion_pytorch_model.safetensors",
        "encoder/config.json",
        "encoder/pytorch_model.bin",
    ]:
        print(hf_hub_download(repo_id="FoivosPar/Arc2Face", filename=filename, local_dir=args.local_dir))

    antelope = os.path.join(args.local_dir, "antelopev2")
    os.makedirs(antelope, exist_ok=True)
    print(
        hf_hub_download(
            repo_id="FoivosPar/Arc2Face",
            filename="arcface.onnx",
            local_dir=antelope,
        )
    )
    print("Place InsightFace antelopev2 ONNX files under", antelope)


if __name__ == "__main__":
    main()
