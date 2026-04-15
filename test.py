import argparse
import ast
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import MVTecDataset, RealIADDataset, get_data_transforms
from models.hybrid_inp import build_inp_former
from utils import evaluation_batch, get_logger, setup_seed


MVTEC_ITEMS = [
    "carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule",
    "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper",
]
VISA_ITEMS = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]
REALIAD_ITEMS = [
    "audiojack", "bottle_cap", "button_battery", "end_cap", "eraser", "fire_hood",
    "mint", "mounts", "pcb", "phone_battery", "plastic_nut", "plastic_plug",
    "porcelain_doll", "regulator", "rolled_strip_base", "sim_card_set", "switch", "tape",
    "terminalblock", "toothbrush", "toy", "toy_brick", "transistor1", "usb",
    "usb_adaptor", "u_block", "vcpill", "wooden_beads", "woodstick", "zipper",
]


def parse_scalar(value: str):
    v = value.strip()
    if v.lower() in ["true", "false"]:
        return v.lower() == "true"
    try:
        return ast.literal_eval(v)
    except Exception:
        return v.strip("\"'")


def load_simple_yaml(path: str):
    if not path or not os.path.exists(path):
        return {}

    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = parse_scalar(value)
    return cfg


def resolve_items(dataset: str):
    if dataset == "MVTec-AD":
        return MVTEC_ITEMS
    if dataset == "VisA":
        return VISA_ITEMS
    if dataset == "Real-IAD":
        return REALIAD_ITEMS
    raise ValueError("Unsupported dataset. Use MVTec-AD, VisA, or Real-IAD.")


def build_test_loader(args, item, data_transform, gt_transform):
    if args.dataset in ["MVTec-AD", "VisA"]:
        test_path = os.path.join(args.data_path, item)
        test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    else:
        test_data = RealIADDataset(
            root=args.data_path,
            category=item,
            transform=data_transform,
            gt_transform=gt_transform,
            phase="test",
        )
    return DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4)


def resolve_checkpoint_path(args, item):
    if args.model_path:
        return args.model_path
    return os.path.join(args.save_dir, args.save_name, item, "model.pth")


def main(args):
    setup_seed(args.seed)
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    items = resolve_items(args.dataset)
    alpha_log = []
    results_all = []

    for item in items:
        model = build_inp_former(args.encoder, args.INP_num, args.device)
        ckpt_path = resolve_checkpoint_path(args, item)
        model.load_state_dict(torch.load(ckpt_path, map_location=args.device), strict=True)
        model.eval()

        test_loader = build_test_loader(args, item, data_transform, gt_transform)
        results = evaluation_batch(
            model,
            test_loader,
            args.device,
            max_ratio=0.01,
            resize_mask=256,
            mode=args.mode,
            prior_path=args.prior_path,
            alpha_mode=args.alpha_mode,
            alpha_fixed=args.alpha_fixed,
            alpha_log_list=alpha_log if args.log_alpha else None,
        )
        results_all.append([item] + list(results))
        print_fn(
            "{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
                item, *results
            )
        )

    mean_metrics = np.mean(np.array([row[1:] for row in results_all], dtype=np.float32), axis=0)
    print_fn(results_all)
    print_fn(
        "Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
            *mean_metrics
        )
    )

    if args.log_alpha and len(alpha_log) > 0:
        alpha_out_dir = os.path.join(args.save_dir, args.save_name)
        os.makedirs(alpha_out_dir, exist_ok=True)
        alpha_path = os.path.join(alpha_out_dir, f"alpha_values_{args.mode}.npy")
        np.save(alpha_path, np.array(alpha_log, dtype=np.float32))
        print_fn(f"Saved alpha logs to: {alpha_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate INP/prior/hybrid inference.")

    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--dataset", type=str, default="MVTec-AD")
    parser.add_argument("--data_path", type=str, required=True)

    parser.add_argument("--save_dir", type=str, default="./saved_results")
    parser.add_argument("--save_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--input_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--INP_num", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--mode", type=str, default="inp", choices=["inp", "prior", "hybrid"])
    parser.add_argument("--prior_path", type=str, default="checkpoints/prior_bank.pt")
    parser.add_argument("--alpha_mode", type=str, default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--alpha_fixed", type=float, default=0.5)
    parser.add_argument("--log_alpha", action="store_true")

    parser.add_argument("--seed", type=int, default=1)

    cli_args = parser.parse_args()
    cfg = load_simple_yaml(cli_args.config)

    for key, value in cfg.items():
        if hasattr(cli_args, key):
            setattr(cli_args, key, value)

    cli_args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    log_name = f"Hybrid-Eval_{cli_args.dataset}_{cli_args.mode}"
    logger = get_logger(log_name, os.path.join(cli_args.save_dir, cli_args.save_name))
    print_fn = logger.info

    main(cli_args)
