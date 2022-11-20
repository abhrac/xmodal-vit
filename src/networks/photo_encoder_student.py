from torch import nn
import timm
from timm.models.vision_transformer import VisionTransformer
import torch


class PhotoEncoder(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super().__init__()
        self.encoder: VisionTransformer = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, output_dim)
        )

    def embedding(self, photo):
        x = self.encoder.patch_embed(photo)
        cls_token = self.encoder.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.encoder.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.encoder.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.encoder.pos_drop(x + self.encoder.pos_embed)
        x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        return x

    def forward_features(self, photo):
        return self.mlp_head(self.encoder.forward_features(photo))

    def classify(self, features):
        return self.mlp_head(features)

    def forward(self, photo):
        return self.classify(self.forward_features(photo))

