import argparse
import os

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from dataset import RealIADDataset, get_data_transforms
from models.hybrid_inp import build_inp_former, extract_encoder_tokens


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


def resolve_items(dataset: str):
    if dataset == "MVTec-AD":
        return MVTEC_ITEMS
    if dataset == "VisA":
        return VISA_ITEMS
    if dataset == "Real-IAD":
        return REALIAD_ITEMS
    raise ValueError("Unsupported dataset. Use MVTec-AD, VisA, or Real-IAD.")


def build_train_loader(dataset: str, data_path: str, item: str, data_transform, gt_transform, batch_size: int):
    if dataset in ["MVTec-AD", "VisA"]:
        train_path = os.path.join(data_path, item, "train")
        train_data = ImageFolder(root=train_path, transform=data_transform)
    else:
        train_data = RealIADDataset(
            root=data_path,
            category=item,
            transform=data_transform,
            gt_transform=gt_transform,
            phase="train",
        )

    return DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    model = build_inp_former(args.encoder, args.INP_num, device)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    items = resolve_items(args.dataset)

    token_buffer = []
    with torch.no_grad():
        for item in items:
            loader = build_train_loader(
                args.dataset,
                args.data_path,
                item,
                data_transform,
                gt_transform,
                args.batch_size,
            )
            for batch in tqdm(loader, desc=f"Collecting tokens ({item})", ncols=80):
                images = batch[0].to(device)
                tokens = extract_encoder_tokens(model, images)
                token_buffer.append(tokens.reshape(-1, tokens.shape[-1]).cpu())

    features = torch.cat(token_buffer, dim=0).numpy().astype(np.float32)
    if args.max_tokens > 0 and features.shape[0] > args.max_tokens:
        idx = np.random.choice(features.shape[0], size=args.max_tokens, replace=False)
        features = features[idx]

    kmeans = MiniBatchKMeans(
        n_clusters=args.K,
        batch_size=min(8192, max(args.K * 16, 512)),
        random_state=args.seed,
        n_init=10,
    )
    kmeans.fit(features)

    prior_bank = torch.from_numpy(kmeans.cluster_centers_).float()
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "prior_bank": prior_bank,
            "K": args.K,
            "encoder": args.encoder,
            "dataset": args.dataset,
            "input_size": args.input_size,
            "crop_size": args.crop_size,
        },
        args.output_path,
    )
    print(f"Saved prior bank to {args.output_path} with shape {tuple(prior_bank.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build prior prototype bank for hybrid INP inference.")
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
