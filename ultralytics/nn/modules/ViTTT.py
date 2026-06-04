# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# ViT^3: Unlocking Test-Time Training in Vision
# Modified by Dongchen Han
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class TTT(nn.Module):
    r""" Test-Time Training block for ViT^3 model.
        - https://arxiv.org/abs/2512.01643
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)
        self.proj = nn.Linear(dim + head_dim, dim)

        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        # --- Forward ---
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig

        # --- Backward ---
        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        # --- Clip gradient (for stability) ---
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        # --- Step ---
        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation='prod'):
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        if implementation == 'conv':
            g = F.conv2d(k.reshape(1, B * C, H, W), e.reshape(B * C, 1, H, W), padding=1, groups=B * C)
            g = g.transpose(0, 1)
        elif implementation == 'prod':
            k = F.pad(k, (1, 1, 1, 1))
            outs = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ys = 1 + dy
                    xs = 1 + dx
                    dot = (k[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                    outs.append(dot)
            g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        else:
            raise NotImplementedError

        # --- Clip gradient (for stability) ---
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)

        # --- Step ---
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x, h, w, rope=None):
        b, n, c = x.shape
        d = c // self.num_heads

        # Prepare q/k/v
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        if rope is not None:
            # 支援動態 H, W
            q1 = rope(q1.reshape(b, h, w, c), h, w).reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = rope(k1.reshape(b, h, w, c), h, w).reshape(b, n, self.num_heads, d).transpose(1, 2)
        else:
            q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
            k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
            
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # Inner training using (k, v)
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation='prod')

        # Apply updated inner module to q
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), w3, padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        # Output proj
        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwc = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, h, w):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = x + self.dwc(x.reshape(x.shape[0], h, w, x.shape[-1]).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding (Dynamic size)."""
    def __init__(self, feature_dim, base=10000):
        super(RoPE, self).__init__()
        self.feature_dim = feature_dim
        self.base = base
        
        # H, W 共用 channel，因此每個維度佔總 channel 數的一半 -> 1/4 (Complex數計算)
        k_max = feature_dim // 4
        assert feature_dim % k_max == 0

        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        self.register_buffer('theta_ks', theta_ks)
        
        self.cached_H = -1
        self.cached_W = -1
        self.cached_rotations = None

    def forward(self, x, H, W):
        device = x.device
        orig_dtype = x.dtype # 紀錄原始的 dtype (例如 float16 或 float32)
        
        # 緩存機制：若輸入形狀或裝置發生變化時，自動更新旋轉矩陣
        if H != self.cached_H or W != self.cached_W or self.cached_rotations is None or self.cached_rotations.device != device:
            y_grid = torch.arange(H, device=device)
            x_grid = torch.arange(W, device=device)
            y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing='ij')
            
            angles = torch.cat([
                y_mesh.unsqueeze(-1) * self.theta_ks,
                x_mesh.unsqueeze(-1) * self.theta_ks
            ], dim=-1)

            rotations_re = torch.cos(angles).unsqueeze(dim=-1)
            rotations_im = torch.sin(angles).unsqueeze(dim=-1)
            rotations = torch.cat([rotations_re, rotations_im], dim=-1)
            self.cached_rotations = torch.view_as_complex(rotations)
            self.cached_H = H
            self.cached_W = W

        # Complex 操作只支援 float32/float64，因此若是 float16 先轉型
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = self.cached_rotations * x
        
        # 將資料拍平並轉型回原始的 dtype (這一步能避免 Float 和 Half 打架的報錯)
        out = torch.view_as_real(pe_x).flatten(-2)
        return out.to(orig_dtype)


class TTTBlock(nn.Module):
    r""" TTT Block. """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.rope = RoPE(feature_dim=dim)
        self.attn = TTT(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        B, L, C = x.shape

        x = x + self.cpe(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # Attention
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, self.rope))

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer. """
    def __init__(self, dim, dim_out, ratio=4.0):
        super().__init__()
        in_channels = dim
        out_channels = dim_out
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
        
        # 動態獲取降採樣過後的新形狀
        _, _, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H_new, W_new


class BasicLayer(nn.Module):
    """ A basic TTT layer for one stage. """
    def __init__(self, dim, dim_out, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            TTTBlock(dim=dim, num_heads=num_heads,
                     mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                     drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, dim_out=dim_out)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W, use_reentrant=False)
            else:
                x = blk(x, H, W)
                
        # 提取尚未降採樣的輸出 (作為這一階的 Feature map 輸出)
        out = x 
        out_H, out_W = H, W
        
        # 進行降採樣，給下個階段使用
        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)
            
        return x, out, H, W, out_H, out_W


class Stem(nn.Module):
    r""" Stem """
    def __init__(self, in_chans=3, embed_dim=96):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        x = self.conv(x)
        # 動態獲取首次卷積後的 H, W
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class ViTTT(nn.Module):
    r""" ViTTT
        A PyTorch impl of : `ViT^3: Unlocking Test-Time Training in Vision`
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 dim=[96, 192, 384, 768], depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = dim[0]
        self.mlp_ratio = mlp_ratio
        
        self.img_size = img_size if isinstance(img_size, int) else img_size[0]
        self.in_chans = in_chans
        self.dim = dim 

        # Stem 已改為動態支持任意輸入大小
        self.patch_embed = Stem(in_chans=in_chans, embed_dim=dim[0])

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=dim[i_layer],
                               dim_out=dim[i_layer + 1] if i_layer < self.num_layers - 1 else None,
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = nn.BatchNorm1d(dim[-1])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # 計算 width_list
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train() 
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            print("Setting width_list to dim as fallback.")
            self.width_list = self.dim
            self.train() 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            if m.groups > 0: 
                fan_out //= m.groups
            if fan_out > 0:
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def forward_features(self, x):
        B = x.shape[0]
        feature_outputs = []
        
        # 獲取動態的 H 和 W
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for i, layer in enumerate(self.layers):
            # 依賴 Block 返回出新的以及舊的 H, W 大小
            x, out, H_new, W_new, out_H, out_W = layer(x, H, W)
            
            C = out.shape[-1]
            
            # 使用當前層對應的 out_H 和 out_W (尚未降採樣前的大小) 來復原張量
            out_spatial = out.reshape(B, out_H, out_W, C).permute(0, 3, 1, 2).contiguous()
            feature_outputs.append(out_spatial)
            
            # 替換為下一層的新形狀
            H, W = H_new, W_new
            
        return feature_outputs

    def forward(self, x):
        features = self.forward_features(x)
        return features

# -------------------------------------------------------------
# Factory Functions
# -------------------------------------------------------------
def h_vittt_tiny(pretrained=False, img_size=224, **kwargs):
    model = ViTTT(img_size=img_size, dim=[64, 128, 320, 512], depths=[1, 3, 9, 4], num_heads=[2, 4, 10, 16], **kwargs)
    return model

def h_vittt_small(pretrained=False, img_size=224, **kwargs):
    model = ViTTT(img_size=img_size, dim=[64, 128, 320, 512], depths=[2, 6, 18, 8], num_heads=[2, 4, 10, 16], **kwargs)
    return model

def h_vittt_base(pretrained=False, img_size=224, **kwargs):
    model = ViTTT(img_size=img_size, dim=[96, 192, 448, 640], depths=[2, 6, 18, 8], num_heads=[3, 6, 14, 20], **kwargs)
    return model


if __name__ == '__main__':
    # 簡單測試用例以確認輸出的正確性以及 width_list
    img_h, img_w = 224, 224
    print("--- Creating ViTTT Tiny model ---")
    model = h_vittt_tiny(img_size=img_h)
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing ViTTT Tiny forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        print("Output feature shapes:")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully.")
    except Exception as e:
        print(f"\nError during testing: {e}")