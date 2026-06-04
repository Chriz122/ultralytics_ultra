import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

# 假設 Conv 模組在您的代碼庫其他地方有定義 (如 Ultralytics YOLO 的 standard Conv)
from .conv import Conv
from .block import C2f, Bottleneck, C3k

class ComLinear(nn.Linear):
    r""" Linear layer for complex number inputs.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__(in_features, out_features, False, device, dtype)

    def forward(self, x):
        x = torch.view_as_real(x).transpose(-2, -1)
        
        # 【修正】: 適配 YOLO 的半精度/混合精度訓練
        # 確保 weight 的 dtype 和 x 匹配，避免 F.linear 因資料型態不一致而報錯
        weight = self.weight.to(x.dtype)
        x = torch.nn.functional.linear(x, weight).transpose(-2, -1)
        
        # cuFFT float16 支援有限，強制轉回 float32 供後續的 fft 運算
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
            
        x = torch.view_as_complex(x.contiguous())
        return x


class CirculantAttention(nn.Module):
    r""" Circulant Attention
    https://arxiv.org/abs/2512.21542
    """
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.qkv = ComLinear(dim, dim * 3)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        
        # 轉換為 (B, H, W, C) 以便於進行 Linear 計算與 FFT
        x = x.permute(0, 2, 3, 1)

        # 這裡的 t 保留原本的資料型態 (例如 float16)
        t = self.gate(x)
        
        # 紀錄原本的 dtype
        orig_dtype = x.dtype
        
        # 【關鍵修正】：強制轉為 float32
        # 以解決 cuFFT 在計算 float16 時只支援 2 的次方尺寸 (如 16, 32) 的問題
        x_f32 = x.to(torch.float32)

        # ---------------- FFT 運算在 float32 空間進行 ----------------
        x_f32 = torch.fft.rfft2(x_f32, dim=(1, 2), norm='ortho')
        qkv = self.qkv(x_f32)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Equation 15 of the paper
        attn = torch.conj(q) * k
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm='ortho')

        # Equation 16 of the paper
        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)
        attn = torch.fft.rfft2(attn, dim=(1, 2))
        x_out = torch.conj(attn) * v
        x_out = torch.fft.irfft2(x_out, s=(h, w), dim=(1, 2), norm='ortho')
        # -------------------------------------------------------------

        # 【關鍵修正】：完成 FFT 後，轉回原來的 dtype (如 float16)，無縫接軌後續網路
        x_out = x_out.to(orig_dtype)

        # Output
        x = x_out * t
        x = self.proj(x)
        
        # 轉回 (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class PCABlock(nn.Module):
    """PCABlock class implementing a Position-Sensitive Attention block for neural networks.
    * Modified to use CirculantAttention instead of v10_Attention *

    Attributes:
        attn (CirculantAttention): Circulant Attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): (Unused in CirculantAttention, kept for API compatibility).
            num_heads (int): (Unused in CirculantAttention, kept for API compatibility).
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__()

        # 【修改】: 替換為代碼1的 CirculantAttention
        self.attn = CirculantAttention(dim=c)
        
        # 保留原本的 FFN 結構 (須確保 Conv 已在代碼環境中導入)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
    
class C2PCA(nn.Module):
    """C2PCA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PCA module, applying attention and feed-forward operations.

    Examples:
        >>> c2pca = C2PCA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2pca(input_tensor)

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        """Initialize C2PCA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PCABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PCA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    
    
class C3k2PCA(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2 module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            attn (bool): Whether to use attention blocks.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PCABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )