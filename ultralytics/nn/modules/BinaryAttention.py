import math
import logging
import io
import os
import time
from collections import defaultdict, deque, OrderedDict
from functools import partial
from copy import deepcopy
from typing import Any, NewType
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function

from .conv import Conv
from .block import C3k, C2f, Bottleneck


""" Quantization """

BinaryTensor = NewType('BinaryTensor', torch.Tensor)  # A type where each element is in {-1, 1}

def binary_sign(x: torch.Tensor) -> BinaryTensor:
    """Return -1 if x < 0, 1 if x >= 0."""
    return x.sign() + (x == 0).type(torch.float) 


class STESign(Function):
    """
    Binarize tensor using sign function.
    Straight-Through Estimator (STE) is used to approximate the gradient of sign function.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor: 
        """
        Return a Sign tensor.

        Args:
            ctx: context
            x: input tensor

        Returns:
            Sign(x) = (x>=0) - (x<0)
            Output type is float tensor where each element is either -1 or 1.
        """
        ctx.save_for_backward(x)
        sign_x = binary_sign(x)
        return sign_x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient using STE.

        Args:
            ctx: context
            grad_output: gradient w.r.t. output of Sign

        Returns:
            Gradient w.r.t. input of the Sign function
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input

binarize = STESign.apply  


class SymQuantizer(Function):
    """
    uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise=False):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip val
        :param num_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            assert input.ndimension() == 4
            max_input = (
                    torch.max(torch.abs(input), dim=-2, keepdim=True)[0]
                    .expand_as(input)
                    .detach()
                )

        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)

        output = torch.round(input * s).div(s + 1e-6)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors 
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

symquantize = SymQuantizer.apply


class BinaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attn_quant=False, attn_bias=False, pv_quant=False, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_quant = attn_quant
        self.attn_bias = attn_bias
        self.pv_quant = pv_quant

        if self.attn_bias: # dense bias
            self.input_size = input_size
            self.num_relative_distance = (2 * input_size[0] - 1) * (2 * input_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(input_size[0])
            coords_w = torch.arange(input_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += input_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += input_size[1] - 1
            relative_coords[:, :, 0] *= 2 * input_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(input_size[0] * input_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _quantize(x):
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        sign = binarize(x)
        return s * sign
    
    @staticmethod
    def _quantize_p(x):
        qmax = 255
        s = 1.0 / qmax 
        q = round_ste(x / s).clamp(0, qmax)
        return s * q
    
    @staticmethod
    def _quantize_v(x, bits=8):
        act_clip_val = torch.tensor([-2.0, 2.0])
        return symquantize(x, act_clip_val, bits, False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if self.attn_quant:

            q = self._quantize(q)
            k = self._quantize(k)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if self.attn_bias:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                            self.input_size[0] * self.input_size[1] + 1,
                            self.input_size[0] * self.input_size[1] + 1, -1)
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                attn = attn + relative_position_bias.unsqueeze(0)

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if self.pv_quant:
                attn = self._quantize_p(attn)
                v = self._quantize_v(v, 8)

        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PBABlock(nn.Module):
    """PSABlock class implementing a Position-Sensitive Attention block for neural networks.
    
    Modified to use BinaryAttention (from Code 1) instead of v10_Attention.

    Attributes:
        attn (BinaryAttention): Multi-head binary attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True, **kwargs) -> None:
        """Initialize the PSABlock.

        Args:
            c (int): Input and output channels.
            attn_ratio (float): Kept for API compatibility (BinaryAttention doesn't use it).
            num_heads (int): Number of attention heads.
            shortcut (bool): Whether to use shortcut connections.
            **kwargs: Extra arguments for BinaryAttention (e.g., attn_quant, pv_quant, attn_bias, input_size).
        """
        super().__init__()

        # 替換為代碼1的 BinaryAttention，傳入維度 dim=c 以及 head 數量
        self.attn = BinaryAttention(dim=c, num_heads=num_heads, **kwargs)
        
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute a forward pass through PSABlock.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor after attention and feed-forward processing.
        """
        B, C, H, W = x.shape
        
        # 1. 調整形狀以適應 BinaryAttention: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x_attn_in = x.flatten(2).transpose(1, 2)
        
        # 2. 進行注意力計算
        x_attn_out = self.attn(x_attn_in)
        
        # 3. 將形狀轉回原本的 2D 空間格式: (B, H*W, C) -> (B, C, H*W) -> (B, C, H, W)
        x_attn_out = x_attn_out.transpose(1, 2).reshape(B, C, H, W)
        
        # 4. 殘差連接 (Shortcut)
        x = x + x_attn_out if self.add else x_attn_out
        x = x + self.ffn(x) if self.add else self.ffn(x)
        
        return x


class C2PBA(nn.Module):
    """C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, **kwargs):
        """Initialize C2PSA module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of PSABlock modules.
            e (float): Expansion ratio.
            **kwargs: Extra arguments passed down to BinaryAttention inside PSABlock.
        """
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        # 在建立 PSABlock 時，可以透過 **kwargs 傳遞 attn_quant=True 等量化參數給 BinaryAttention
        self.m = nn.Sequential(*(
            PBABlock(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64), **kwargs) 
            for _ in range(n)
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process the input tensor through a series of PSA blocks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after processing.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C3k2PBA(C2f):
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
                PBABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )