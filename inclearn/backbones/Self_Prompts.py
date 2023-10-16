import torch
import torch.nn as nn
from inclearn.lib.utils import logger
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model
from functools import partial
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.vision_transformer import _init_vit_weights
from timm.models.vision_transformer import VisionTransformer
from typing import Union
from collections import OrderedDict


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        res = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x += res
        return x


class SelfPrompt(nn.Module):
    def __init__(self, extra_token_nums=1, *args, **kwargs):
        # super(SelfPrompt, self).__init__()
        # self.extra_tokens = ExtraTokens(embedding_dim=self.embed_dim)
        self.num_extra_tokens = 1
        self.features_dim = self.embed_dim
        self.pos_embed_new = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + extra_token_nums, self.embed_dim))
        self.current_prompts = None
        self.extra_token = None

    def add_extra_tokens(self, extra_tokens):
        self.extra_token = extra_tokens

    def set_batch_prompts(self, batch_targets):
        # this function is used for adding prompts for a batch.
        # input: targets for one batch
        batch_targets = set(batch_targets)
        self.current_prompts = [self.extra_tokens.tokens[i] for i in batch_targets]

    def set_global_prompts(self, all_p=True, specific=None):
        if all_p:
            self.current_prompts = list(self.extra_tokens.tokens.items())
        else:
            self.set_batch_prompts(specific)

    def concat_prompts(self, x, prompts):
        """
        For images in a batch, each image have different prompt
        either from manually specification or auto generation

        Args:
            x: image patches, with shape of [B, N, C]
            prompts: prompts to be merged, with shape of [B, N_prompts, C]
        Returns:
            tensor with shape of [B, N+N_prompts, C]

        """
        assert x.shape[0] == prompts.shape[0]
        assert x.shape[-1] == prompts.shape[-1]
        return torch.cat((prompts, x), dim=1)

    def forward_features(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, extra_tokens=None, only_feat=True, return_immediate_feat=False, **kwargs):
        x = self.forward_features(x, extra_tokens=extra_tokens, return_immediate_feat=return_immediate_feat)
        if only_feat or return_immediate_feat:
            return x
        else:
            return self.head(x)

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "blocks":
            model = self.blocks
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable

        if not trainable:
            model.eval()
        else:
            model.train()
        return self

    def set_backbone_frozen(self):
        raise NotImplementedError

    def reset_pos_embed(self, num):
        self.pos_embed_new = torch.nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + num, self.embed_dim).cuda())


class SelfPromptDeit(SelfPrompt, VisionTransformer):
    def __init__(self, extra_token_nums=2, immediate_layer=9, immediate_after_norm=False, replace_cls=False, *args,
                 **kwargs):
        VisionTransformer.__init__(self, *args, **kwargs)
        SelfPrompt.__init__(self, extra_token_nums=extra_token_nums)  # 2 = class_token (1) + prompt (1)
        logger.info(f'replace_cls layer: {replace_cls}')
        self.replace_cls = replace_cls
        logger.info(f'Immediate layer: {immediate_layer}')
        self.immediate_layer = immediate_layer

        self.immediate_after_norm = immediate_after_norm

        if self.immediate_after_norm:
            assert self.immediate_layer == len(self.blocks) - 1
            logger.info('Using immediate output after norm.')
        else:
            assert 0 <= self.immediate_layer <= len(self.blocks) - 1

    def get_deep_params(self, layer=None):
        params = []
        if layer is None:
            layer = self.immediate_layer
        for idx, b in enumerate(self.blocks):  # type:Union[int, nn.Module]
            if idx > layer:
                params.extend(list(b.parameters()))
        return params

    def set_backbone_frozen(self, num_layer=None):
        self.patch_embed.training = False
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad_(False)

        # self.blocks.training = False
        flag = False
        for idx, b in enumerate(self.blocks):
            if num_layer is not None:
                if idx > num_layer:
                    b.train()
                    b.requires_grad_(True)
                    flag = True
                    continue
            logger.warning(f'Set block No.{idx} frozen.')
            b.eval()
            b.requires_grad_(False)
        if not flag:
            logger.warning(f'No blocks are Training.')
            self.norm.eval()
            self.norm.requires_grad_(False)

    def reset_deep_layers(self, layer=None):
        if layer is None:
            layer = self.immediate_layer
        for idx, b in enumerate(self.blocks):  # type:Union[int, nn.Module]
            if idx > layer:
                logger.warning(f'Reset block NO.[{idx}] with [{sum(p.numel() for p in b.parameters())}] parameters.')
                b.apply(self._init_weights)

    def forward(self, x, extra_tokens=None, only_feat=True, return_immediate_feat=False, shortcut=False,
                break_immediate=False):
        if shortcut:
            x = self.forward_features_shotcut(x, extra_tokens=extra_tokens)
        else:
            x = self.forward_features(x, return_immediate_feat=return_immediate_feat, break_immediate=break_immediate)

        if only_feat or return_immediate_feat:  # always true
            return x
        else:
            assert False
            return self.head(x)

    def forward_features(self, x, return_immediate_feat=False, only_immediate=False, break_immediate=True):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        immediate = None
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx == self.immediate_layer:
                immediate = x
                if break_immediate:
                    break
        if not break_immediate:
            x = self.norm(x)
            if self.immediate_after_norm:
                immediate = x
        if only_immediate:
            return immediate
        elif return_immediate_feat:
            return x[:, 0], immediate
        else:
            return x[:, 0]

    def forward_features_shotcut(self, x, extra_tokens=None):
        if extra_tokens is not None:
            if self.replace_cls:
                x_tmp = x[:, 1:, ...]
                x = self.concat_prompts(extra_tokens, x_tmp)
            else:
                x_tmp = x[:, 1:, ...]
                x_tmp = self.concat_prompts(extra_tokens, x_tmp)
                x = torch.cat((x[:, 0, ...].unsqueeze(1), x_tmp), dim=1)
        for idx, blk in enumerate(self.blocks):
            if idx > self.immediate_layer:
                x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class ExtraTokens(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.tokens = torch.nn.ParameterDict()
        self.embedding_dim = embedding_dim
        # self.task_indices = {}
        self.current_task = -1
        self.num_tokens = 1

    def set_frozen(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_first_tokens(self, all_tokens):
        if all_tokens is not None:
            logger.warning(f'To overwrite origin tokens:\n\t {self.tokens}'
                           f'\n new tokens\n\t {all_tokens}')
            self.tokens['0'] = all_tokens

    # def add_tokens(self, classes, num_tokens, is_new_task=True, rand=False):
    #     assert num_tokens == 1, 'currently supports num_tokens == 1 per class.'
    #     classes = list(classes)
    #     for cls in classes:
    #         if rand:
    #             self.tokens[str(cls)] = nn.Parameter(torch.rand(self.embedding_dim).cuda())
    #         else:
    #             self.tokens[str(cls)] = nn.Parameter(torch.zeros(self.embedding_dim).cuda())
    #     if is_new_task:
    #         self.task_indices[self.current_task] = classes
    #         self.current_task += 1

    def add_tokens(self, num_tokens, is_new_task=True, rand=False):
        logger.info('Added new Tokens into APG.')
        if is_new_task:
            if len(self.tokens):
                # all_tokens = torch.cat(list(self.tokens.values()))
                # all_tokens_mean = all_tokens.mean(dim=0)
                # all_tokens_std = all_tokens.std(dim=0)
                # new_tokens_weight = [torch.normal(mean=all_tokens_mean, std=all_tokens_std) for _ in range(num_tokens)]
                # new_tokens = nn.Parameter(torch.stack(new_tokens_weight).cuda())
                new_tokens = nn.Parameter(torch.rand(num_tokens, self.embedding_dim).cuda())
            else:
                new_tokens = nn.Parameter(torch.zeros(num_tokens, self.embedding_dim).cuda())
                torch.nn.init.normal_(new_tokens, mean=0, std=.02)
            self.current_task += 1
            self.tokens[str(self.current_task)] = new_tokens

    def add_tokens_spec(self, classes, tokens, is_new_task=True):
        classes = list(classes)
        for cls, token in zip(classes, tokens):
            self.tokens[str(cls)] = nn.Parameter(token)
        if is_new_task:
            self.task_indices[self.current_task] = classes
            self.current_task += 1

    def get_old_tokens(self):
        if self.current_task == 0:
            return []
        # old_task_classes = self.task_indices[self.current_task - 1]
        # old_tokens = [self.tokens[str(i)] for i in old_task_classes]
        old_tokens = self.tokens[str(self.current_task - 1)]
        return [old_tokens]

    def get_all_tokens(self, return_dict=False):
        if return_dict:
            return self.tokens
        all_tokens = list(self.tokens.values())
        return torch.cat(all_tokens).cuda().reshape(1, -1, all_tokens[0].shape[-1])

    def reset_tokens(self):
        for w in self.tokens.values():
            nn.init.kaiming_normal_(w)


class AdaptivePromptGenerator(nn.Module):
    def __init__(self, embedding_dim, MLP_num=2, num_heads=4, attn_drop=0., proj_drop=0., more_prompts=1, attn_depth=1,
                 use_update_tokens=False, cross_attn_wo_x=False, residual=True):
        super().__init__()
        logger.info(f'MLP_num: [{MLP_num}],'
                    f'Num_heads [{num_heads}],'
                    f'attn_drop [{attn_drop}],'
                    f'proj_drop [{proj_drop}]'
                    f'cross_attn_wo_x [{cross_attn_wo_x}],'
                    f'residual [{residual}]')
        self.embedding_dim = embedding_dim
        self.attn_depth = attn_depth
        self.use_update_tokens = use_update_tokens
        self.MLPs_in = Mlp(in_features=embedding_dim, out_features=embedding_dim, act_layer=nn.GELU)
        self.MLPs_out = Mlp(in_features=embedding_dim, out_features=embedding_dim, act_layer=nn.GELU)
        self.num_heads = num_heads
        self.cross_attn_wo_x = cross_attn_wo_x
        self.attn = nn.ModuleDict(OrderedDict({f'attn{i}': MyAttention(embedding_dim,
                                                                       num_heads=self.num_heads,
                                                                       qkv_bias=True,
                                                                       attn_drop=attn_drop,
                                                                       proj_drop=proj_drop,
                                                                       more_prompts=more_prompts,
                                                                       cross_attn_wo_x=self.cross_attn_wo_x,
                                                                       residual=residual) for i in
                                               range(attn_depth)}))
        self.norms = nn.ModuleDict(
            OrderedDict({f'norm{i}': nn.LayerNorm(self.embedding_dim) for i in range(attn_depth - 1)}))
        logger.info(f'current_attn_layers:{self.attn}')

        # self.attn = MyAttention(embedding_dim, num_heads=self.num_heads, qkv_bias=True, attn_drop=attn_drop,
        #                         proj_drop=proj_drop, more_prompts=more_prompts)
        # self.middle = nn.Sequential(*[Mlp(embedding_dim, embedding_dim // 2, embedding_dim) for _ in range(MLP_num)])
        self.all_tokens = ExtraTokens(embedding_dim)
        # self.norm1 = nn.LayerNorm(self.embedding_dim)
        # self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.norm3 = nn.LayerNorm(self.embedding_dim)
        self.norm4 = nn.LayerNorm(self.embedding_dim)
        # self.norm5 = nn.LayerNorm(self.embedding_dim)
        self.apply(_init_vit_weights)

    def forward(self, img_feat, all_tokens=True, specific=None, return_attn=False):
        # assert all_tokens or (not all_tokens and specific is not None)
        tokens = self.all_tokens.get_all_tokens() if all_tokens else specific
        # tokens = self.token_MLP(tokens)
        # img_feat = self.norm1(img_feat)
        # tokens = self.norm2(tokens)
        q = self.MLPs_in(img_feat)

        q_new = self.norm3(q)
        attn = torch.tensor([]).cuda()
        if self.use_update_tokens:
            for i in range(self.attn_depth):
                if i == 0:
                    q_new, attn_new, other_tokens = self.attn[f'attn{i}'](q_new, tokens, return_all_tokens=True)
                else:
                    q_new, attn_new, other_tokens = self.attn[f'attn{i}'](q_new, other_tokens, return_all_tokens=True)

                if i != self.attn_depth - 1:
                    q_new = self.norms[f'norm{i}'](q_new)

                attn = torch.cat((attn, attn_new), dim=0)
        else:
            for i in range(self.attn_depth):
                q_new, attn_new = self.attn[f'attn{i}'](q_new, tokens)
                if i != self.attn_depth - 1:
                    q_new = self.norms[f'norm{i}'](q_new)
                attn = torch.cat((attn, attn_new), dim=0)
        # q_new = self.middle(q_new)
        # attn = None
        # q_new = q_new + q
        # q_new1 = q_new
        # q_new = self.norm4(q_new)
        q_new = self.MLPs_out(q_new)
        # q_new = self.norm4(q_new)
        # q_new = q_new + q_new1
        if return_attn:
            return q_new, attn
        else:
            return q_new


class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., more_prompts=1,
                 cross_attn_wo_x=False, residual=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.more_prompts = more_prompts  # output #self.more_prompts prompts.
        if self.more_prompts > 1:
            logger.warning(f'Output prompts #{self.more_prompts} instead of one.')
        else:
            logger.info(f'Output Prompts #{self.more_prompts}')
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim * self.more_prompts, bias=qkv_bias)
        self.k = nn.Linear(dim, dim * self.more_prompts, bias=qkv_bias)
        self.v = nn.Linear(dim, dim * self.more_prompts, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cross_attn_wo_x = cross_attn_wo_x
        self.residual = residual

    def forward(self, x, prompts, return_all_tokens=False):
        res_x = x
        B, N, C = x.shape
        assert N == 1
        _, N_prompts, C_prompts = prompts.shape
        assert C == C_prompts  # assert with same dim

        if self.cross_attn_wo_x:
            x_with_prompts = prompts.expand(B, N_prompts, C)
            k_v_length = N_prompts
        else:
            x_with_prompts = torch.cat((x, prompts.expand(B, N_prompts, C)), dim=1)
            k_v_length = N_prompts + N
        if return_all_tokens:
            res_x = res_x[:, 0, ...]
            assert self.more_prompts == 1
            q = self.q(x_with_prompts).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                               C // self.num_heads).permute(0, 2, 1, 3)
        else:
            q = self.q(x).unsqueeze(1).reshape(B, N, self.more_prompts * self.num_heads,
                                               C // self.num_heads).permute(0, 2, 1, 3)

        k = self.k(x_with_prompts).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                           C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x_with_prompts).reshape(B, k_v_length, self.more_prompts * self.num_heads,
                                           C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1))
        raw_attn = attn
        attn = attn.softmax(dim=-1)
        attn = attn * self.scale

        attn = self.attn_drop(attn)

        if return_all_tokens:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, N_prompts + self.more_prompts, C)
        else:
            x_cls = (attn @ v).transpose(1, 2).reshape(B, self.more_prompts, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        if return_all_tokens:
            other_tokens = x_cls[:, 1:, ...]
            x_cls = x_cls[:, 0, ...]
            if self.residual:
                x_cls += res_x
            x_cls = x_cls.unsqueeze(1)
            return x_cls, raw_attn, other_tokens
        else:
            if self.residual:
                x_cls += res_x
            return x_cls, raw_attn


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 0, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


@register_model
def SelfPrompt_tiny(pretrained=False, **kwargs):
    model = SelfPromptCait(
        img_size=224, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/XXS24_224.pth",
            map_location="cpu", check_hash=True
        )
        checkpoint_no_module = {}
        for k in model.state_dict().keys():
            checkpoint_no_module[k] = checkpoint["model"]['module.' + k]

        model.load_state_dict(checkpoint_no_module)

    return model


@register_model
def SelfPromptDeit_base(pretrained=False, **kwargs):
    model = SelfPromptDeit(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model


@register_model
def SelfPromptDeit_small(pretrained=False, **kwargs):
    model = SelfPromptDeit(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model


@register_model
def SelfPromptDeit_mytiny(pretrained=False, extra_token_nums=2, immediate_layer=9, **kwargs):
    model = SelfPromptDeit(extra_token_nums=extra_token_nums,
                           immediate_layer=immediate_layer,
                           patch_size=16, embed_dim=256, depth=12, num_heads=8, mlp_ratio=4, qkv_bias=True,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = _cfg()
    return model


@register_model
def SelfPromptCait_tiny(pretrained=False, **kwargs):
    model = SelfPromptCait(
        img_size=224, patch_size=16, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2, **kwargs)

    model.default_cfg = _cfg()
    return model


if __name__ == '__main__':
    atten = MyAttention(dim=36, num_heads=2, more_prompts=2)

    x = torch.rand(16, 1, 36)

    prompt = torch.rand(1, 50, 36)
    out = atten(x, prompt)
    print(out)
    net = SelfPromptDeit_mytiny()
    params = 0
    for param in net.parameters():
        params += param.numel()
    print(params)
