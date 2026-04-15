import argparse

import numpy as np
import torch

from prototypes.build_prior import main as build_prior_main


DATASET_ALIASES = {
    "mvtec": "MVTec-AD",
    "mvtec-ad": "MVTec-AD",
    "visa": "VisA",
    "real-iad": "Real-IAD",
    "realiad": "Real-IAD",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build prior prototype bank.")
    parser.add_argument("--dataset", type=str, default="MVTec-AD")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--INP_num", type=int, default=6)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--max_tokens", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="checkpoints/prior_bank.pt")
    args = parser.parse_args()

    dataset_key = args.dataset.strip().lower()
    args.dataset = DATASET_ALIASES.get(dataset_key, args.dataset)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    build_prior_main(args)
