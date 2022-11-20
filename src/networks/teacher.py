"""
Acknowledgements:
1. https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
2. https://github.com/rishikksh20/CrossViT-pytorch
"""

import torch
from einops.layers.torch import Rearrange
from torch import nn

from networks.module import Attention, PreNorm, FeedForward, CrossAttention
from networks.photo_encoder import PhotoEncoder
from networks.sketch_encoder import SketchEncoder


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class XMA(nn.Module):

    def __init__(self, dim=192, dim_head=64, cross_attn_depth=1, cross_attn_heads=3, dropout=0.):
        super().__init__()

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
                PreNorm(dim,
                        CrossAttention(dim, heads=cross_attn_heads, dim_head=dim_head, dropout=dropout)),
                nn.Linear(dim, dim),
                nn.Linear(dim, dim),
                PreNorm(dim,
                        CrossAttention(dim, heads=cross_attn_heads, dim_head=dim_head, dropout=dropout)),
            ]))

    def forward(self, x_branch_1, x_branch_2):
        for f_12, g_21, cross_attn_s, f_21, g_12, cross_attn_l in self.cross_attn_layers:
            branch_1_class = x_branch_1[:, 0]
            x_branch_1 = x_branch_1[:, 1:]
            branch_2_class = x_branch_2[:, 0]
            x_branch_2 = x_branch_2[:, 1:]

            cal_q = f_21(branch_2_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_branch_1), dim=1)
            cal_out = cal_q + cross_attn_l(cal_qkv)
            cal_out = g_12(cal_out)
            x_branch_2 = torch.cat((cal_out, x_branch_2), dim=1)

            cal_q = f_12(branch_1_class.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_branch_2), dim=1)
            cal_out = cal_q + cross_attn_s(cal_qkv)
            cal_out = g_21(cal_out)
            x_branch_1 = torch.cat((cal_out, x_branch_1), dim=1)

        return x_branch_1, x_branch_2


def vit_repr(encoder, x):
    return encoder.embedding(x)


class ModalityFusionNetwork(nn.Module):
    def __init__(self, image_size, channels, patch_size=16, feature_dim=192, cross_attn_depth=1,
                 enc_depth=3, heads=3, pool='cls', dropout=0., emb_dropout=0.,
                 encoder_backbone='vit_base_patch16_224'):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_branch_1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, feature_dim),
        )

        self.to_patch_embedding_branch_2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, feature_dim),
        )

        self.pos_embedding_branch_1 = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim))
        self.cls_token_branch_1 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.dropout_branch_1 = nn.Dropout(emb_dropout)

        self.pos_embedding_branch_2 = nn.Parameter(torch.randn(1, num_patches + 1, feature_dim))
        self.cls_token_branch_2 = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.dropout_branch_2 = nn.Dropout(emb_dropout)

        self.x_modal_transformers = nn.ModuleList([])
        for _ in range(enc_depth):
            self.x_modal_transformers.append(
                XMA(dim=feature_dim,
                    dim_head=feature_dim // heads,
                    cross_attn_depth=cross_attn_depth, cross_attn_heads=heads,
                    dropout=dropout))

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, 1)
        )

        self.sigmoid = nn.Sigmoid()

        transformer_enc_branch_1 = PhotoEncoder(num_classes=125, encoder_backbone=encoder_backbone)
        transformer_enc_branch_2 = SketchEncoder(num_classes=125, encoder_backbone=encoder_backbone)

        self.transformer_enc_branch_1 = transformer_enc_branch_1
        self.transformer_enc_branch_2 = transformer_enc_branch_2

    def forward(self, image_1, image_2):
        return self.classify(self.compose_reprs(self.forward_features(image_1, image_2)))

    def repr_branch_1(self, image):
        encoder = self.transformer_enc_branch_1
        return vit_repr(encoder, image)

    def repr_branch_2(self, image):
        encoder = self.transformer_enc_branch_2
        return vit_repr(encoder, image)

    def cross_modal_embedding(self, x_branch_1, x_branch_2):
        for x_modal_transformer in self.x_modal_transformers:
            x_branch_1, x_branch_2 = x_modal_transformer(x_branch_1, x_branch_2)

        x_branch_1 = x_branch_1.mean(dim=1) if self.pool == 'mean' else x_branch_1[:, 0]
        x_branch_2 = x_branch_2.mean(dim=1) if self.pool == 'mean' else x_branch_2[:, 0]

        return x_branch_1, x_branch_2

    def forward_features(self, image_1, image_2):
        x_branch_1 = self.repr_branch_1(image_1)
        x_branch_2 = self.repr_branch_2(image_2)

        return self.cross_modal_embedding(x_branch_1, x_branch_2)

    @staticmethod
    def compose_reprs(x_branches):
        return torch.cat(x_branches, dim=1)

    def classify(self, x_repr):
        return self.sigmoid(self.mlp_head(x_repr))
