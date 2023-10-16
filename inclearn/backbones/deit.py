# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import logging

import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from inclearn.lib.logger import LOGGER

log = LOGGER
logger = log.LOGGER

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384', 'deit_tiny_prompt_16_224'
]


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


class PromptDeit(VisionTransformer):

    def __init__(self, prompt_numbers=10, insert_mode='shallow', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_numbers_per_task = prompt_numbers
        self.insert_mode = insert_mode
        assert self.insert_mode in ['shallow', 'deep', 'baseline'], f'Got {self.insert_mode}'
        # self.Prompt = nn.ParameterList([]).cuda()
        self.Prompt = {i: [] for i in range(len(self.blocks))}
        logger.info(f'Current Prompt Config:\n\tInsert Mode: {self.insert_mode}\n\t'
                    f'Prompt_numbers_per_task: {self.prompt_numbers_per_task}\n\t')
        # self.cls_token =
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens, self.embed_dim).cuda())

    # @property
    # def num_tokens(self):
    #     return 1 + len(self.Prompt)

    def add_prompt(self):
        # new_prompt.requires_grad = True
        if self.insert_mode == 'baseline':
            logger.info(f'Baseline method, not adding any prompts.')
            return
        new_prompt = nn.Parameter(torch.zeros(1, self.prompt_numbers_per_task, self.embed_dim).cuda(),
                                  requires_grad=True)
        print(new_prompt.device)
        self.Prompt[0].append(new_prompt)
        self.reset_pos_embed()
        logger.info(f'added prompt[0]')
        if self.insert_mode == 'deep':
            for idx in range(1, len(self.blocks)):
                current_prompt = nn.Parameter(torch.zeros(1, self.prompt_numbers_per_task, self.embed_dim).cuda(),
                                              requires_grad=True)
                self.Prompt[idx].append(current_prompt)
                logger.info(f'added prompt[{idx}]')

    def reset_pos_embed(self):
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_tokens + len(
            self.Prompt[0]) * self.prompt_numbers_per_task, self.embed_dim).cuda())

    def set_old_prompt_frozen(self):
        for b_idx, prompt in self.Prompt.items():
            if not len(prompt):
                continue
            for p in prompt[:-1]:
                p.requires_grad = False
            prompt[-1].requires_grad = True

    def set_backbone_frozen(self, num_layer=None):
        self.patch_embed.training = False
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad_(False)

        # self.blocks.training = False
        for idx, b in enumerate(self.blocks):
            if num_layer is not None:
                if idx > num_layer:
                    b.train()
                    b.requires_grad_(True)
                    continue
            b.eval()
            b.requires_grad_(False)

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        prompt = [self.Prompt[0][i].expand(x.shape[0], -1, -1) for i in range(len(self.Prompt[0]))]
        x = torch.cat((cls_token, *prompt, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        # x = self.blocks(x)
        for idx, blk in enumerate(self.blocks):
            if self.insert_mode == 'shallow':
                x = blk(x)
            elif self.insert_mode == 'deep':
                if idx == 0:
                    x = blk(x)
                else:
                    prompt = [self.Prompt[idx][i].expand(x.shape[0], -1, -1) for i in range(len(self.Prompt[idx]))]
                    prompt = torch.cat(prompt, dim=1)
                    x[:, self.num_tokens: len(self.Prompt[0]) * self.prompt_numbers_per_task + self.num_tokens,
                    :] = prompt
                    x = blk(x)

        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        return x


@register_model
def deit_tiny_prompt_16_224(pretrained=False, prompt_numbers=50, insert_mode='shallow', **kwargs):
    model = PromptDeit(
        prompt_numbers=prompt_numbers, insert_mode=insert_mode,
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    # if pretrained:
    #     checkpoint = torch.hub.load_state_dict_from_url(
    #         url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
    #         map_location="cpu", check_hash=True
    #     )
    #     model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    # net = deit_tiny_prompt_16_224(insert_mode='shallow').cuda()
    # net.add_prompt()
    # a = torch.rand(16, 3, 224, 224).cuda()
    # out = net.forward_features(a)
    # net.set_backbone_frozen()
    # net.add_prompt()
    # out = net.forward_features(a)
    # label = torch.rand(16, 192).cuda()
    # loss = torch.nn.MSELoss()
    # a = loss(out, label)
    # a.backward()
    pass
    net = deit_small_patch16_224()
    total_params = sum(p.numel() for p in net.parameters())
    print('总参数个数：{}'.format(total_params))
