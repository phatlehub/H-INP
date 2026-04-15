from functools import partial

import torch
import torch.nn as nn

from models import vit_encoder
from models.uad import INP_Former
from models.vision_transformer import Aggregation_Block, Mlp, Prototype_Block


def infer_model_dims(encoder_name: str):
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    if "small" in encoder_name:
        return 384, 6, target_layers
    if "base" in encoder_name:
        return 768, 12, target_layers
    if "large" in encoder_name:
        return 1024, 16, [4, 6, 8, 10, 12, 14, 16, 18]
    raise ValueError("Architecture not in small, base, large.")


def build_inp_former(encoder_name: str, inp_num: int, device: str):
    """Build baseline INP-Former architecture used in train/test scripts."""
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    encoder = vit_encoder.load(encoder_name)
    embed_dim, num_heads, target_layers = infer_model_dims(encoder_name)

    bottleneck = nn.ModuleList([Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.0)])
    inp_tokens = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim)) for _ in range(1)])

    inp_extractor = nn.ModuleList([
        Aggregation_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
    ])

    inp_guided_decoder = nn.ModuleList([
        Prototype_Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
        )
        for _ in range(8)
    ])

    model = INP_Former(
        encoder=encoder,
        bottleneck=bottleneck,
        aggregation=inp_extractor,
        decoder=inp_guided_decoder,
        target_layers=target_layers,
        remove_class_token=True,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
        prototype_token=inp_tokens,
    )
    return model.to(device)


def extract_encoder_tokens(model: INP_Former, images: torch.Tensor) -> torch.Tensor:
    """Extract encoder patch tokens [B, N, C] with the same logic as INP-Former."""
    x = model.encoder.prepare_tokens(images)
    en_list = []
    for i, blk in enumerate(model.encoder.blocks):
        if i > model.target_layers[-1]:
            continue
        with torch.no_grad():
            x = blk(x)
        if i in model.target_layers:
            en_list.append(x)

    if model.remove_class_token:
        en_list = [e[:, 1 + model.encoder.num_register_tokens :, :] for e in en_list]

    x = model.fuse_feature(en_list)
    return x
