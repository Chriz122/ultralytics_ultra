# Copyright (c) ByteDance Inc. All rights reserved.
from functools import partial
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_
from torch import nn

__all__ =['nextvit_small', 'nextvit_base', 'nextvit_large']

NORM_EPS = 1e-5

class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


# ==========================================
# Attention Residuals (AttnRes) Components
# ==========================================
class RMSNorm2d(nn.Module):
    def __init__(self, dim, eps=NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        # x shape: [..., C, H, W]
        norm = torch.sqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return (x / norm) * self.weight

class AttnResOp(nn.Module):
    """
    Attention Residuals for 2D spatial features.
    Computes depth-wise softmax attention over all previous blocks in the current stage.
    """
    def __init__(self, dim):
        super().__init__()
        # Initialized to zero to ensure equal-weight average at the start of training (Paper Section 5)
        self.w = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm2d(dim)

    def forward(self, V_list):
        # V_list contains[h_1, f_1(h_1), f_2(h_2), ...]
        V = torch.stack(V_list)  # Shape: [L, B, C, H, W]
        K = self.norm(V)
        
        # Spatial-aware depth-wise dot-product attention
        # w: [C], K:[L, B, C, H, W] -> logits: [L, B, H, W]
        logits = torch.einsum('c, l b c h w -> l b h w', self.w, K)
        attn_weights = logits.softmax(dim=0)
        
        # Weighted sum of historical features
        # attn_weights: [L, B, H, W], V:[L, B, C, H, W] -> h: [B, C, H, W]
        h = torch.einsum('l b h w, l b c h w -> b c h w', attn_weights, V)
        return h
# ==========================================


class MHCA(nn.Module):
    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class NCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0,
                 drop=0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False
        
        # AttnRes modules for NCB
        self.mhca_attn_res = AttnResOp(out_channels)
        self.mlp_attn_res = AttnResOp(out_channels)

    def forward(self, x, V_list):
        is_identity = isinstance(self.patch_embed.conv, nn.Identity) and isinstance(self.patch_embed.avgpool, nn.Identity)
        
        if not is_identity:
            # Dimension changed (Stage boundary). Reset AttnRes state list.
            x_emb = self.patch_embed(x)
            V_list = [x_emb]
        else:
            # Identity mapping, preserve existing V_list from previous layer
            pass

        # === 1. MHCA Sub-layer with AttnRes ===
        h_mhca = self.mhca_attn_res(V_list)
        attn_out = self.mhca(h_mhca)
        attn_out = self.attention_path_dropout(attn_out)
        V_list.append(attn_out)

        # === 2. MLP Sub-layer with AttnRes ===
        h_mlp = self.mlp_attn_res(V_list)
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            h_mlp_norm = self.norm(h_mlp)
        else:
            h_mlp_norm = h_mlp
            
        mlp_out = self.mlp(h_mlp_norm)
        mlp_out = self.mlp_path_dropout(mlp_out)
        V_list.append(mlp_out)

        # The accumulated output for this block is the sum of all components
        x_accum = sum(V_list)
        return x_accum, V_list


class E_MHSA(nn.Module):
    def __init__(self, dim, out_dim=None, head_dim=32, qkv_bias=True, qk_scale=None,
                 attn_drop=0, proj_drop=0., sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio ** 2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1d(kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1d(dim, eps=NORM_EPS)
        self.is_bn_merged = False

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(B, N, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.transpose(1, 2)
            x_ = self.sr(x_)
            if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose(1, 2)
            k = self.k(x_)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x_)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        else:
            k = self.k(x)
            k = k.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 3, 1)
            v = self.v(x)
            v = v.reshape(B, -1, self.num_heads, int(C // self.num_heads)).permute(0, 2, 1, 3)
        attn = (q @ k) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NTB(nn.Module):
    def __init__(
            self, in_channels, out_channels, path_dropout, stride=1, sr_ratio=1,
            mlp_ratio=2, head_dim=32, mix_block_ratio=0.75, attn_drop=0, drop=0,
    ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2d, eps=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels, stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(self.mhsa_out_channels, head_dim=head_dim, sr_ratio=sr_ratio,
                             attn_drop=attn_drop, proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

        # AttnRes modules for internal varying-dimension branches of NTB
        self.mhsa_attn_res = AttnResOp(self.mhsa_out_channels)
        self.mhca_attn_res = AttnResOp(self.mhca_out_channels)
        self.mlp_attn_res = AttnResOp(self.out_channels)

    def forward(self, x, V_list):
        # === 1. E_MHSA Branch ===
        # NTB dynamically splits channels here, requiring its own localized accumulation context
        x_mhsa_emb = self.patch_embed(x)
        V_list_mhsa = [x_mhsa_emb]
        
        h_mhsa = self.mhsa_attn_res(V_list_mhsa)
        
        B, C, H, W = h_mhsa.shape
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm1(h_mhsa)
        else:
            out = h_mhsa
            
        out = rearrange(out, "b c h w -> b (h w) c")
        out = self.mhsa_path_dropout(self.e_mhsa(out))
        mhsa_out_img = rearrange(out, "b (h w) c -> b c h w", h=H)
        V_list_mhsa.append(mhsa_out_img)
        x_mhsa_accum = sum(V_list_mhsa)

        # === 2. MHCA Branch ===
        x_mhca_emb = self.projection(x_mhsa_accum)
        V_list_mhca =[x_mhca_emb]
        
        h_mhca = self.mhca_attn_res(V_list_mhca)
        mhca_out = self.mhca_path_dropout(self.mhca(h_mhca))
        V_list_mhca.append(mhca_out)
        x_mhca_accum = sum(V_list_mhca)

        # === 3. MLP Branch ===
        x_cat = torch.cat([x_mhsa_accum, x_mhca_accum], dim=1)
        V_list_out = [x_cat]
        
        h_mlp = self.mlp_attn_res(V_list_out)
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out3 = self.norm2(h_mlp)
        else:
            out3 = h_mlp
            
        mlp_out = self.mlp_path_dropout(self.mlp(out3))
        V_list_out.append(mlp_out)
        x_out_accum = sum(V_list_out)

        return x_out_accum, V_list_out


class NextViT(nn.Module):
    def __init__(self, stem_chs, depths, path_dropout, attn_drop=0, drop=0, num_classes=1000,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75,
                 use_checkpoint=False):
        super(NextViT, self).__init__()
        self.use_checkpoint = use_checkpoint

        self.stage_out_channels = [[96] * (depths[0]),
                                   [192] * (depths[1] - 1) + [256],[384, 384, 384, 384, 512] * (depths[2] // 5),
                                   [768] * (depths[3] - 1) + [1024]]

        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1] - 1) + [NTB],[NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
                                  [NCB] * (depths[3] - 1) + [NTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
        )
        input_channel = stem_chs[-1]
        
        # Modules are collected in nn.ModuleList for custom forward traversal
        self.features = nn.ModuleList()
        idx = 0
        dpr =[x.item() for x in torch.linspace(0, path_dropout, sum(depths))]
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                stride = 2 if (strides[stage_id] == 2 and block_id == 0) else 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel, output_channel, stride=stride, path_dropout=dpr[idx + block_id],
                                drop=drop, head_dim=head_dim)
                    self.features.append(layer)
                elif block_type is NTB:
                    layer = NTB(input_channel, output_channel, path_dropout=dpr[idx + block_id], stride=stride,
                                sr_ratio=sr_ratios[stage_id], head_dim=head_dim, mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop, drop=drop)
                    self.features.append(layer)
                input_channel = output_channel
            idx += numrepeat

        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)
        self.stage_out_idx = [sum(depths[:idx + 1]) - 1 for idx in range(len(depths))]
        
        # Determine width list securely dynamically
        self.width_list =[i.size(1) for i in self.forward(torch.randn(1, 3, 640, 640))]
        self._initialize_weights()

    def _initialize_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, AttnResOp):
                # Enforce pseudo-query weight is 0 exactly at start
                nn.init.constant_(m.w, 0.0)

    def forward(self, x):
        res =[]
        x = self.stem(x)
        
        # State tracker for cross-layer Attention Residual Accumulation Context
        V_list = [x]

        for idx, layer in enumerate(self.features):
            if self.use_checkpoint:
                # Custom forward wrapper for checkpoint packing tensors cleanly
                def custom_forward(module):
                    def fn(x_in, *v_args):
                        out_x, out_v = module(x_in, list(v_args))
                        return tuple([out_x] + out_v)
                    return fn
                
                outs = checkpoint.checkpoint(custom_forward(layer), x, *V_list)
                x = outs[0]
                V_list = list(outs[1:])
            else:
                x, V_list = layer(x, V_list)
                
            if idx in self.stage_out_idx:
                res.append(x)
                
        res[-1] = self.norm(res[-1])
        return res

def update_weight(model_dict, weight_dict):
    idx, temp_dict = 0, {}
    for k, v in weight_dict.items():
        if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
            temp_dict[k] = v
            idx += 1
    model_dict.update(temp_dict)
    print(f'loading weights... {idx}/{len(model_dict)} items')
    return model_dict

def nextvit_small_attnres(weights=''):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 10, 3], path_dropout=0.1)
    if weights:
        pretrained_weight = torch.load(weights)['model']
        model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
    return model


def nextvit_base_attnres(weights=''):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 20, 3], path_dropout=0.2)
    if weights:
        pretrained_weight = torch.load(weights)['model']
        model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
    return model


def nextvit_large_attnres(weights=''):
    model = NextViT(stem_chs=[64, 32, 64], depths=[3, 4, 30, 3], path_dropout=0.2)
    if weights:
        pretrained_weight = torch.load(weights)['model']
        model.load_state_dict(update_weight(model.state_dict(), pretrained_weight))
    return model