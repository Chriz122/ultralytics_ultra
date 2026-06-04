#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from pathlib import Path
import numpy as np


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, B):
    C = windows.shape[2]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    return x


def ct_dewindow(ct, H, W, window_size):
    bs = ct.shape[0]
    C = ct.shape[-1]
    ct2 = ct.view(bs, H // window_size, W // window_size, window_size, window_size, C)
    ct2 = ct2.permute(0, 5, 1, 3, 2, 4).reshape(bs, C, H * W).transpose(1, 2)
    return ct2


def ct_window(ct, H, W, window_size):
    bs = ct.shape[0]
    C = ct.shape[-1]
    ct2 = ct.transpose(1, 2).view(bs, C, H // window_size, window_size, W // window_size, window_size)
    ct2 = ct2.permute(0, 2, 4, 3, 5, 1).reshape(bs, -1, C)
    return ct2


def _load_state_dict(module, state_dict, strict=False, logger=None):
    unexpected_keys =[]
    all_missing_keys = []
    err_msg =[]

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    
    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys =[
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            pass


def _load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class PosEmbMLPSwinv2D(nn.Module):
    def __init__(self,
                 window_size,
                 pretrained_window_size,
                 num_heads, seq_length,
                 ct_correct=False,
                 no_log=False):
        super().__init__()
        self.window_size = window_size
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads
        self.ct_correct = ct_correct
        self.no_log = no_log
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # 保留緩衝區以防載入權重時出現 KeyError
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
        self.register_buffer("relative_coords_table", relative_coords_table)
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        relative_bias = torch.zeros(1, num_heads, seq_length, seq_length)
        self.register_buffer("relative_bias", relative_bias)

    def forward(self, input_tensor, local_window_size, cur_win_h=None, cur_win_w=None):
        if cur_win_h is None: cur_win_h = self.window_size[0]
        if cur_win_w is None: cur_win_w = self.window_size[1]
        
        relative_coords_h = torch.arange(-(cur_win_h - 1), cur_win_h, dtype=torch.float32, device=input_tensor.device)
        relative_coords_w = torch.arange(-(cur_win_w - 1), cur_win_w, dtype=torch.float32, device=input_tensor.device)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).permute(1, 2, 0).contiguous().unsqueeze(0)
            
        if self.pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (cur_win_h - 1 + 1e-8)
            relative_coords_table[:, :, :, 1] /= (cur_win_w - 1 + 1e-8)

        if not self.no_log:
            relative_coords_table *= 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        # ⭐ 重要修正：將座標張量轉為與輸入相同的 dtype，避免 FP16 (Half) 驗證/訓練時引發 dtype 不匹配的錯誤
        relative_coords_table = relative_coords_table.to(input_tensor.dtype)

        relative_position_bias_table = self.cpb_mlp(relative_coords_table).view(-1, self.num_heads)
        
        coords_h = torch.arange(cur_win_h, device=input_tensor.device)
        coords_w = torch.arange(cur_win_w, device=input_tensor.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += cur_win_h - 1
        relative_coords[:, :, 1] += cur_win_w - 1
        relative_coords[:, :, 0] *= 2 * cur_win_w - 1
        relative_position_index = relative_coords.sum(-1)

        relative_position_bias = relative_position_bias_table[relative_position_index.view(-1)].view(
            cur_win_h * cur_win_w, cur_win_h * cur_win_w, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        n_global_feature = input_tensor.shape[2] - cur_win_h * cur_win_w
        if n_global_feature > 0 and self.ct_correct:
            step_for_ct = cur_win_h / (n_global_feature**0.5 + 1)
            seq_length = int(n_global_feature ** 0.5)
            indices =[]
            for i in range(seq_length):
                for j in range(seq_length):
                    ind = (i+1)*step_for_ct*cur_win_h + (j+1)*step_for_ct
                    indices.append(int(ind))

            top_part = relative_position_bias[:, indices, :]
            lefttop_part = relative_position_bias[:, indices, :][:, :, indices]
            left_part = relative_position_bias[:, :, indices]
            
        if n_global_feature > 0:
            relative_position_bias = torch.nn.functional.pad(relative_position_bias, (n_global_feature, 0, n_global_feature, 0)).contiguous()
            if self.ct_correct:
                relative_position_bias = relative_position_bias * 0.0
                relative_position_bias[:, :n_global_feature, :n_global_feature] = lefttop_part
                relative_position_bias[:, :n_global_feature, n_global_feature:] = top_part
                relative_position_bias[:, n_global_feature:, :n_global_feature] = left_part

        pos_emb = relative_position_bias.unsqueeze(0)
        
        input_tensor = input_tensor + pos_emb
        return input_tensor


class PosEmbMLPSwinv1D(nn.Module):
    def __init__(self,
                 dim,
                 rank=2,
                 seq_length=4,
                 conv=False):
        super().__init__()
        self.rank = rank
        if not conv:
            self.cpb_mlp = nn.Sequential(nn.Linear(self.rank, 512, bias=True),
                                         nn.ReLU(),
                                         nn.Linear(512, dim, bias=False))
        else:
            self.cpb_mlp = nn.Sequential(nn.Conv1d(self.rank, 512, 1,bias=True),
                                         nn.ReLU(),
                                         nn.Conv1d(512, dim, 1,bias=False))
        self.conv = conv
        # 保留緩衝區以防 KeyError
        relative_bias = torch.zeros(1, seq_length, dim)
        self.register_buffer("relative_bias", relative_bias)

    def forward(self, input_tensor, ct_h=None, ct_w=None):
        seq_length = input_tensor.shape[1] if not self.conv else input_tensor.shape[2]
        
        if self.rank == 1:
            relative_coords_h = torch.arange(0, seq_length, device=input_tensor.device, dtype=input_tensor.dtype)
            relative_coords_h -= seq_length//2
            relative_coords_h /= (seq_length//2 + 1e-8)
            pos_emb = self.cpb_mlp(relative_coords_h.unsqueeze(0).unsqueeze(2))
        else:
            if ct_h is not None and ct_w is not None:
                h, w = ct_h, ct_w
            else:
                h = w = int(seq_length**0.5)
            relative_coords_h = torch.arange(0, h, device=input_tensor.device, dtype=input_tensor.dtype)
            relative_coords_w = torch.arange(0, w, device=input_tensor.device, dtype=input_tensor.dtype)
            relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij')).contiguous().unsqueeze(0)
            relative_coords_table[:, 0, :, :] -= h // 2
            relative_coords_table[:, 0, :, :] /= (h // 2 + 1e-8)
            relative_coords_table[:, 1, :, :] -= w // 2
            relative_coords_table[:, 1, :, :] /= (w // 2 + 1e-8)
            
            if not self.conv:
                pos_emb = self.cpb_mlp(relative_coords_table.flatten(2).transpose(1,2))
            else:
                pos_emb = self.cpb_mlp(relative_coords_table.flatten(2))
            
        input_tensor = input_tensor + pos_emb
        return input_tensor


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x


class Downsample(nn.Module):
    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, in_dim=64, dim=96):
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in[int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, global_feature=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x, global_feature


class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 resolution=0,
                 seq_length=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pos_emb_funct = PosEmbMLPSwinv2D(window_size=[resolution, resolution],
                                              pretrained_window_size=[resolution, resolution],
                                              num_heads=num_heads,
                                              seq_length=seq_length)

        self.resolution = resolution

    def forward(self, x, h=None, w=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        cur_win_h = h if h is not None else self.resolution
        cur_win_w = w if w is not None else self.resolution

        attn = self.pos_emb_funct(attn, self.resolution ** 2, cur_win_h, cur_win_w)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAT(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1.,
                 window_size=7,
                 last=False,
                 layer_scale=None,
                 ct_size=1,
                 do_propagation=False):
        super().__init__()
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=window_size**2)
        self.norm1 = norm_layer(dim)
        cr_tokens_per_window = ct_size**2 if sr_ratio > 1 else 0
        cr_tokens_total = cr_tokens_per_window*sr_ratio*sr_ratio
        self.cr_window = ct_size
        self.attn = WindowAttention(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    resolution=window_size,
                                    seq_length=window_size**2 + cr_tokens_per_window)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = window_size

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, resolution=int(cr_tokens_total**0.5),
                seq_length=cr_tokens_total)

            self.hat_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.hat_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.hat_pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=cr_tokens_total)
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.upsampler = nn.Upsample(size=window_size, mode='nearest')

        self.last = last
        self.do_propagation = do_propagation

    def forward(self, x, carrier_tokens, num_windows_h=None, num_windows_w=None):
        B, T, N = x.shape
        ct = carrier_tokens
        x = self.pos_embed(x)

        if self.sr_ratio > 1 and ct is not None:
            Bg, Ng, Hg = ct.shape

            ct_h = num_windows_h * self.cr_window
            ct_w = num_windows_w * self.cr_window

            ct = ct_dewindow(ct, ct_h, ct_w, self.cr_window)
            ct = self.hat_pos_embed(ct, ct_h, ct_w)
            
            ct1 = self.hat_attn(self.hat_norm1(ct), ct_h, ct_w)
            ct = ct + self.hat_drop_path(self.gamma1 * ct1)
            
            ct2 = self.hat_mlp(self.hat_norm2(ct))
            ct = ct + self.hat_drop_path(self.gamma2 * ct2)

            ct = ct_window(ct, ct_h, ct_w, self.cr_window)
            ct = ct.reshape(x.shape[0], -1, N)
            x = torch.cat((ct, x), dim=1)

        x1 = self.attn(self.norm1(x))
        x = x + self.drop_path(self.gamma3 * x1)
        
        x2 = self.mlp(self.norm2(x))
        x = x + self.drop_path(self.gamma4 * x2)

        if self.sr_ratio > 1 and ct is not None:
            ctr, x = x.split([x.shape[1] - self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
            ct = ctr.reshape(Bg, Ng, Hg)
            if self.last and self.do_propagation:
                ctr_image_space = ctr.transpose(1, 2).reshape(x.shape[0], N, self.cr_window, self.cr_window)
                upsampled = self.upsampler(ctr_image_space.to(dtype=torch.float32))
                upsampled = upsampled.flatten(2).transpose(1, 2).to(dtype=x.dtype)
                x = x + self.gamma1 * upsampled
                
        return x, ct


class TokenInitializer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 ct_size=1):
        super().__init__()
        self.ct_size = ct_size
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        x = self.pos_embed(x)
        B, C, H, W = x.shape
        num_windows_h = H // self.window_size
        num_windows_w = W // self.window_size
        out_h = num_windows_h * self.ct_size
        out_w = num_windows_w * self.ct_size
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (out_h, out_w))
        
        B, C, H_out, W_out = x.shape
        ct = x.view(B, C, num_windows_h, self.ct_size, num_windows_w, self.ct_size)
        ct = ct.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, C)
        return ct


class FasterViTLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 num_heads,
                 window_size,
                 ct_size=1,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 only_local=False,
                 hierarchy=True,
                 do_propagation=False
                 ):
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(dim=dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layer_scale=layer_scale_conv)
                for i in range(depth)])
            self.transformer_block = False
        else:
            sr_ratio = input_resolution // window_size if not only_local else 1
            self.blocks = nn.ModuleList([
                HAT(dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    sr_ratio=sr_ratio,
                    window_size=window_size,
                    last=(i == depth-1),
                    layer_scale=layer_scale,
                    ct_size=ct_size,
                    do_propagation=do_propagation,
                    )
                for i in range(depth)])
            self.transformer_block = True
            
        self.downsample = None if not downsample else Downsample(dim=dim)
        if len(self.blocks) and not only_local and input_resolution // window_size > 1 and hierarchy and not self.conv:
            self.global_tokenizer = TokenInitializer(dim,
                                                     input_resolution,
                                                     window_size,
                                                     ct_size=ct_size)
            self.do_gt = True
        else:
            self.do_gt = False

        self.window_size = window_size

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 動態補零 (Padding) 計算以適應特徵圖不被 window_size 整除的輸入大小
        if self.transformer_block:
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            
            if pad_b > 0 or pad_r > 0:
                x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
                
        Hp, Wp = x.shape[2], x.shape[3]
        num_windows_h = Hp // self.window_size if self.transformer_block else None
        num_windows_w = Wp // self.window_size if self.transformer_block else None
        
        ct = self.global_tokenizer(x) if self.do_gt else None
        
        if self.transformer_block:
            x = window_partition(x, self.window_size)
            
        for bn, blk in enumerate(self.blocks):
            if isinstance(blk, HAT):
                x, ct = blk(x, ct, num_windows_h, num_windows_w)
            else:
                x, ct = blk(x, ct)
                
        if self.transformer_block:
            x = window_reverse(x, self.window_size, Hp, Wp, B)
            if Hp > H or Wp > W:
                x = x[:, :, :H, :W] # 裁切回原始解析度
                
        out = x # 特徵圖(下採樣前)
        if self.downsample is not None:
            x = self.downsample(x)
        return out, x


class FasterViT(nn.Module):
    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 ct_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 layer_norm_last=False,
                 hat=[False, False, True, False],
                 do_propagation=False,
                 **kwargs):
        super().__init__()
        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr =[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        if hat is None: hat = [True, ]*len(depths)
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False
            level = FasterViTLayer(dim=int(dim * 2 ** i),
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   ct_size=ct_size,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   conv=conv,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   downsample=(i < 3),
                                   layer_scale=layer_scale,
                                   layer_scale_conv=layer_scale_conv,
                                   input_resolution=int(2 ** (-2 - i) * resolution),
                                   only_local=not hat[i],
                                   do_propagation=do_propagation)
            self.levels.append(level)
        self.norm = LayerNorm2d(num_features) if layer_norm_last else nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

        self.width_list =[]
        try:
            self.eval() 
            dummy_input = torch.randn(1, in_chans, resolution, resolution)
            with torch.no_grad():
                 features = self.forward(dummy_input)
            self.width_list =[f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to embed_dims as fallback.")
            self.width_list =[int(dim * 2 ** i) for i in range(len(depths))]
            self.train() 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        outs =[]
        for level in self.levels:
            out, x = level(x)
            outs.append(out)
        outs[-1] = self.norm(outs[-1])
        return outs
    
    def forward_head(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        return features
    
    def _load_state_dict(self, 
                         pretrained, 
                         strict: bool = False):
        _load_checkpoint(self, 
                         pretrained, 
                         strict=strict)


@register_model
def faster_vit_0(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[2, 3, 6, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=64,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.2,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def faster_vit_1(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[1, 3, 8, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=80,
                      in_dim=32,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.2,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def faster_vit_2(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[3, 3, 8, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=96,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.2,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def faster_vit_3(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[3, 3, 12, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=128,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.3,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      do_propagation=True,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def faster_vit_4(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[3, 3, 12, 5],
                      num_heads=[4, 8, 16, 32],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=196,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.3,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      layer_norm_last=False,
                      do_propagation=True,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def faster_vit_5(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[3, 3, 12, 5],
                      num_heads=[4, 8, 16, 32],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=320,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.3,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      layer_norm_last=False,
                      do_propagation=True,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def faster_vit_6(pretrained=False, resolution=224, **kwargs):
    model = FasterViT(depths=[3, 3, 16, 8],
                      num_heads=[4, 8, 16, 32],
                      window_size=[7, 7, 7, 7],
                      ct_size=2,
                      dim=320,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=resolution,
                      drop_path_rate=0.5,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      layer_norm_last=False,
                      do_propagation=True,
                      hat=[False, False, True, False],
                      **kwargs)
    model.default_cfg = _cfg()
    return model


