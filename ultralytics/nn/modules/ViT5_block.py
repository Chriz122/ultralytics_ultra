import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat

# 依賴保留
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath

try:
    from flash_attn import flash_attn_qkvpacked_func
except ImportError:
    pass

def broadcat(freqss, dim=-1):
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    assert len(shape_lens) == 1, 'freqss must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims =[(i, val) for i, val in enumerate(dims) if i != dim]
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim=dim)


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, pt_seq_len=14, custom_freqs=None, freqs_for='lang', theta=10000, max_freq=10, num_freqs=1):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * math.pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        self.pt_seq_len = pt_seq_len
        self.register_buffer("freqs", freqs)

    def forward(self, x, H=None, W=None): 
        # 修改: 動態支援 YOLO 的特徵圖 H, W，並去除 numpy 與硬派的 .cuda()
        if H is None or W is None:
            H = W = int(math.sqrt(x.shape[1]))
            
        t_h = torch.arange(H, device=x.device).float() / H * self.pt_seq_len
        t_w = torch.arange(W, device=x.device).float() / W * self.pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t_h, self.freqs)
        freqs_w = torch.einsum('..., f -> ... f', t_w, self.freqs)
        
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r=2) 
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r=2) 
        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim=-1)

        freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1])
        return x * freqs_cos + rotate_half(x) * freqs_sin


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., flash=True,
                 rope_size=0, rope_reg_size=0, num_registers=0, reg_theta=10000, qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.flash = flash
        # 注意: YOLO 沒有分類 token，所以 num_registers 設為 0
        self.rope = VisionRotaryEmbedding(head_dim//2, rope_size) if rope_size > 0 else None

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)

        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        if self.rope is not None:
            # 移除了針對 CLS token 的切片，全部視為空間特徵
            q = self.rope(q, H, W)
            k = self.rope(k, H, W)
        
        # 【修改這裡】動態判斷是否滿足 FlashAttention 的嚴格條件 (CUDA 且為半精度)
        use_flash = self.flash and q.is_cuda and (q.dtype in[torch.float16, torch.bfloat16])
        
        if use_flash:
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_qkvpacked_func(qkv).reshape(B, N, C)
        else:
            # 降級為常規的 Attention 運算 (YOLO CPU 初始化 FP32 時會走這裡)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block=Attention, Mlp_block=Mlp, init_values=1e-4,
                 flash=True, rope_size=0, rope_reg_size=0, reg_theta=10000, num_registers=0, qk_norm=False, layer_scale=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, flash=flash,
            rope_size=rope_size, rope_reg_size=rope_reg_size, num_registers=num_registers, qk_norm=qk_norm, reg_theta=reg_theta)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, H=None, W=None):
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H=H, W=W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H=H, W=W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT5Block(nn.Module):
    """為 YOLO 打造的外殼，處理 (B, C, H, W) 與 (B, H*W, C) 轉換"""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.c = c2
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, 0) if c1 != c2 else nn.Identity()

        # 動態分配 num_heads，確保 head_dim (c2 // num_heads) 固定在 32
        # 這能完美相容 FlashAttention 規範，並保證 RoPE 運算的維度正常
        num_heads = max(1, c2 // 32)
        if c2 % num_heads != 0:
            num_heads = 1

        self.blocks = nn.ModuleList([
            Block(
                dim=c2, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, 
                norm_layer=partial(RMSNorm, eps=1e-6), rope_size=14, 
                num_registers=0, qk_norm=True, flash=True, layer_scale=True
            ) for _ in range(n)
        ])

    def forward(self, x):
        x = self.cv1(x)
        B, C, H, W = x.shape
        # 轉為 Transformer 所需的序列格式
        x = x.flatten(2).transpose(1, 2)
        
        for blk in self.blocks:
            x = blk(x, H=H, W=W)
            
        # 還原為 YOLO 所需的特徵圖格式
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x

        