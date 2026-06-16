import logging
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .conv import Conv
from .block import C3k, SpatialAttnRes

USE_FLASH_ATTN = False
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")


# ---------------------------------------------------------
# 新增: ViT^3 的 MLP 模塊 (含 Depth-wise Convolution 輔助特徵提取)
# ---------------------------------------------------------
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


class TTTAttn(nn.Module):
    """
    Test-Time Training (TTT) module.
    融合了 ViT^3 中 Test-Time Training 的線性全局特徵捕捉能力。
    
    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
    """

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads

        # --- TTT 初始化 ---
        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=True)
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
        # TTT 內部學習迴圈: Simplified SwiGLU update
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig

        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)

        w1, w2 = w1 - lr * g1, w2 - lr * g2
        return w1, w2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0, implementation='prod'):
        # TTT 內部學習迴圈: 3x3 depth-wise convolution update
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

        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        w = w.repeat(B, 1, 1, 1) - lr * g
        return w

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # ===============================
        #  ViT^3 Test-Time Training (TTT)
        # ===============================
        d = self.head_dim
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # Prepare q/k/v for TTT branches
        qkv = self.qkv(x_flat)
        q1, k1, v1, q2, k2, v2 = torch.split(qkv, [C, C, C, d, d, d], dim=-1)

        q1 = q1.reshape(B, N, self.num_heads, d).transpose(1, 2)
        k1 = k1.reshape(B, N, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(B, N, self.num_heads, d).transpose(1, 2)
        
        q2 = q2.reshape(B, H, W, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(B, H, W, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(B, H, W, d).permute(0, 3, 1, 2)

        # 進行 test-time inner training
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3, implementation='prod')

        # 使用更新後的參數推論 Query
        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(B, N, C)
        
        x2 = F.conv2d(q2.reshape(1, B * d, H, W), w3, padding=1, groups=B * d)
        x2 = x2.reshape(B, d, N).transpose(1, 2)

        # 融合輸出
        out = torch.cat([x1, x2], dim=-1)
        out = self.proj(out)
        
        return out.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------
# 核心創新: TTT 全局指導 Area-Attention 局部模塊
# ---------------------------------------------------------
class Hybrid_TTT_Area_Attn(nn.Module):
    """
    Hybrid Test-Time Training & Area-Attention Module.
    透過將 Channel 分半，並使用 TTT 的全局特徵生成 Spatial Gate 來「調製/更新」 Area-Attention 的局部特徵。
    """
    def __init__(self, dim, num_heads, area=1):
        super().__init__()
        assert num_heads >= 2, "num_heads must be at least 2 for Hybrid Attention."
        
        self.dim = dim
        self.num_heads = num_heads
        self.area = area
        
        # 將維度對半切：一半給 TTT (看全圖)，一半給 Area (看局部細節)
        self.dim_ttt = dim // 2
        self.dim_area = dim - self.dim_ttt
        
        self.heads_ttt = num_heads // 2
        self.heads_area = num_heads - self.heads_ttt
        
        self.head_dim_ttt = self.dim_ttt // self.heads_ttt
        self.head_dim_area = self.dim_area // self.heads_area

        # ========================================
        # 1. TTT 模塊初始化 (負責全局 Global Context)
        # ========================================
        self.qkv_ttt = nn.Linear(self.dim_ttt, self.dim_ttt * 3 + self.head_dim_ttt * 3, bias=True)
        self.w1 = nn.Parameter(torch.zeros(1, self.heads_ttt, self.head_dim_ttt, self.head_dim_ttt))
        self.w2 = nn.Parameter(torch.zeros(1, self.heads_ttt, self.head_dim_ttt, self.head_dim_ttt))
        self.w3 = nn.Parameter(torch.zeros(self.head_dim_ttt, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)
        
        equivalent_head_dim = 9
        self.scale = equivalent_head_dim ** -0.5

        # ========================================
        # 2. Area-Attention 模塊初始化 (負責局部細節)
        # ========================================
        all_head_dim_area = self.head_dim_area * self.heads_area
        self.qk_area = Conv(self.dim_area, all_head_dim_area * 2, 1, act=False)
        self.v_area = Conv(self.dim_area, all_head_dim_area, 1, act=False)
        self.pe_area = Conv(all_head_dim_area, self.dim_area, 5, 1, 2, g=self.dim_area, act=False)

        # ========================================
        # 3. 融合與調製層 (Modulation & Fusion)
        # ========================================
        # 用於將 TTT 的全局特徵轉換為指導 Area 的 Spatial Gate (0~1)
        self.ttt_gate = nn.Sequential(
            nn.Conv2d(self.dim_ttt, self.dim_area, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 將拼接後的特徵映射回原始 dim
        self.proj = nn.Linear(self.dim_ttt + self.head_dim_ttt + self.dim_area, dim)

    def inner_train_simplified_swiglu(self, k, v, w1, w2, lr=1.0):
        # TTT 內部學習更新 (SwiGLU 分支)
        z1 = k @ w1
        z2 = k @ w2
        sig = F.sigmoid(z2)
        a = z2 * sig
        e = - v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * a)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - lr * g1, w2 - lr * g2

    def inner_train_3x3dwc(self, k, v, w, lr=1.0):
        # TTT 內部學習更新 (3x3 Depth-wise Conv 分支)
        B, C, H, W = k.shape
        e = - v / float(v.shape[2] * v.shape[3]) * self.scale
        k_pad = F.pad(k, (1, 1, 1, 1))
        outs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ys, xs = 1 + dy, 1 + dx
                dot = (k_pad[:, :, ys: ys + H, xs: xs + W] * e).sum(dim=(-2, -1))
                outs.append(dot)
        g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3)
        g = g / (g.norm(dim=[-2, -1], keepdim=True) + 1.0)
        return w.repeat(B, 1, 1, 1) - lr * g

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        
        # 將輸入切分為兩路
        x_ttt, x_area = torch.split(x, [self.dim_ttt, self.dim_area], dim=1)

        # ========================================
        # 分支 A: TTT 全局注意力 (Global Context)
        # ========================================
        d_ttt = self.head_dim_ttt
        x_ttt_flat = x_ttt.flatten(2).transpose(1, 2)
        
        qkv = self.qkv_ttt(x_ttt_flat)
        q1, k1, v1, q2, k2, v2 = torch.split(qkv, [self.dim_ttt, self.dim_ttt, self.dim_ttt, d_ttt, d_ttt, d_ttt], dim=-1)

        q1 = q1.reshape(B, N, self.heads_ttt, d_ttt).transpose(1, 2)
        k1 = k1.reshape(B, N, self.heads_ttt, d_ttt).transpose(1, 2)
        v1 = v1.reshape(B, N, self.heads_ttt, d_ttt).transpose(1, 2)
        
        q2 = q2.reshape(B, H, W, d_ttt).permute(0, 3, 1, 2)
        k2 = k2.reshape(B, H, W, d_ttt).permute(0, 3, 1, 2)
        v2 = v2.reshape(B, H, W, d_ttt).permute(0, 3, 1, 2)

        # 執行 Test-Time Training 在線更新
        w1, w2 = self.inner_train_simplified_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3x3dwc(k2, v2, self.w3)

        x1 = ((q1 @ w1) * F.silu(q1 @ w2)).transpose(1, 2).reshape(B, N, self.dim_ttt)
        x2 = F.conv2d(q2.reshape(1, B * d_ttt, H, W), w3, padding=1, groups=B * d_ttt).reshape(B, d_ttt, N).transpose(1, 2)
        
        # TTT 輸出的全圖特徵
        out_ttt_flat = torch.cat([x1, x2], dim=-1) # (B, N, dim_ttt + d_ttt)
        out_ttt_spatial = x1.transpose(1, 2).reshape(B, self.dim_ttt, H, W)

        # ========================================
        # 分支 B: Area 局部注意力 (Local Details)
        # ========================================
        qk = self.qk_area(x_area).flatten(2).transpose(1, 2)
        v = self.v_area(x_area)
        pp = self.pe_area(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, self.dim_area * 2)
            v = v.reshape(B * self.area, N // self.area, self.dim_area)
            B_a, N_a = B * self.area, N // self.area
        else:
            B_a, N_a = B, N

        q, k = qk.split([self.dim_area, self.dim_area], dim=2)
        
        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B_a, N_a, self.heads_area, self.head_dim_area)
            k = k.view(B_a, N_a, self.heads_area, self.head_dim_area)
            v = v.view(B_a, N_a, self.heads_area, self.head_dim_area)
            out_area_tmp = flash_attn_func(q.contiguous().half(), k.contiguous().half(), v.contiguous().half()).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B_a, self.heads_area, self.head_dim_area, N_a)
            k = k.transpose(1, 2).view(B_a, self.heads_area, self.head_dim_area, N_a)
            v = v.transpose(1, 2).view(B_a, self.heads_area, self.head_dim_area, N_a)
            
            attn = (q.transpose(-2, -1) @ k) * (self.head_dim_area ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            
            out_area_tmp = (v @ attn.transpose(-2, -1)).permute(0, 3, 1, 2)

        # 恢復形狀並加上 Area-Attention 的位置編碼
        out_area_flat = out_area_tmp.reshape(B, N, self.dim_area)
        out_area_spatial = out_area_flat.transpose(1, 2).reshape(B, self.dim_area, H, W)
        out_area_spatial = out_area_spatial + pp

        # ========================================
        # 4. TTT 更新與調製 Area-Attention
        # ========================================
        # 核心亮點：用 TTT 提取出的全局特徵，產生一個 sigmoid mask 來強化/弱化局部特徵
        global_gate = self.ttt_gate(out_ttt_spatial)
        out_area_modulated = out_area_spatial * global_gate

        # ========================================
        # 5. 輸出特徵融合
        # ========================================
        out_area_flat_modulated = out_area_modulated.flatten(2).transpose(1, 2)
        
        # Concat [TTT 全局特徵, 被 TTT 強化的 Area 局部特徵]
        out_concat = torch.cat([out_ttt_flat, out_area_flat_modulated], dim=-1)
        
        # 降維並恢復為空間維度 (B, C, H, W)
        out = self.proj(out_concat).transpose(1, 2).reshape(B, self.dim, H, W)
        return out


class TTTBlock(nn.Module):
    """
    TTTBlock class implementing a Attention block based on Test-Time Training (ViT^3).
    
    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio. Defaults to 1.2;
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, drop=0., drop_path=0.):
        super().__init__()
        self.attn = TTTAttn(dim, num_heads=num_heads)
        
        # 引入 ViT^3 的 CPE 與專屬 MLP
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Conditional Positional Encoding (CPE)
        x = x + self.cpe(x)
        
        # TTT Attention with LayerNorm
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.drop_path(self.attn(x_norm))
        
        # FFN with LayerNorm
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm2(x_flat)
        x_mlp = self.mlp(x_norm, H, W).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.drop_path(x_mlp)
        
        return x


class HybridABlock(nn.Module):
    """
    ABlock class implementing the Hybrid TTT & Area-Attention block.
    """
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, drop=0., drop_path=0.):
        super().__init__()
        
        # 條件位置編碼 (Conditional Positional Encoding) 幫助 Transformer 獲取平移不變性
        self.cpe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Hybrid_TTT_Area_Attn(dim, num_heads=num_heads, area=area)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # CPE
        x = x + self.cpe(x)
        
        # Attention
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm1(x_flat).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.drop_path(self.attn(x_norm))
        
        # FFN
        x_flat = x.flatten(2).transpose(1, 2)
        x_norm = self.norm2(x_flat)
        x_mlp = self.mlp(x_norm, H, W).transpose(1, 2).reshape(B, C, H, W)
        x = x + self.drop_path(x_mlp)
        
        return x

# ---------------------------------------------------------
# 修改: A2C2f 整合 Attention Residuals 及 TTT 強化
# ---------------------------------------------------------
class TTTC2f_AttnRes(nn.Module):  
    """
    TTTC2f module with residual enhanced feature extraction using TTT/TTTBlock blocks. Also known as R-ELAN
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of TTTlock be a multiple of 32."

        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(TTTBlock(c_, num_heads, mlp_ratio) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

        # 針對深度方向引入 AttnRes
        self.attn_res_layers = nn.ModuleList([SpatialAttnRes(c_) for _ in range(n)])

    def forward(self, x):
        """Forward pass through R-ELAN layer with Attention Residual aggregation."""
        cv1_out = self.cv1(x)
        y = [cv1_out]
        
        # 記錄主幹上的歷史特徵狀態，用於計算 AttnRes
        past_states = [cv1_out]

        for i, m in enumerate(self.m):
            # 1. 取出並進行深度方向自適應加權聚合
            h_in = self.attn_res_layers[i](past_states)
            # 2. 進行 ABlock / TTT 等網絡層運算
            h_out = m(h_in)
            # 3. 推進歷史狀態
            past_states.append(h_out)
            y.append(h_out)

        # 殘差與併發分支整合
        if self.gamma is not None:
            return x + (self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))).to(x.dtype)
        
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------
# A2C2f 整合 Attention Residuals 與 Hybrid TTT 模塊
# ---------------------------------------------------------
class HybridA2C2f_TTT_AttnRes(nn.Module):  
    """
    A2C2f module enhanced with Hybrid TTT & Area-Attention and Attention Residuals.
    """
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        num_heads = c_ // 32
        # 防止 Head 數量小於 2，導致 Hybrid Attention 拆分錯誤
        if num_heads < 2:
            num_heads = 2

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01  
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(HybridABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

        self.attn_res_layers = nn.ModuleList([SpatialAttnRes(c_) for _ in range(n)])

    def forward(self, x):
        cv1_out = self.cv1(x)
        y = [cv1_out]
        past_states = [cv1_out]

        for i, m in enumerate(self.m):
            h_in = self.attn_res_layers[i](past_states)
            h_out = m(h_in)
            past_states.append(h_out)
            y.append(h_out)

        if self.gamma is not None:
            return x + (self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))).to(x.dtype)
        
        return self.cv2(torch.cat(y, 1))


# ---------------------------------------------------------
# A2C2f 整合 Hybrid TTT 模塊
# ---------------------------------------------------------
class HybridA2C2f_TTT(nn.Module):  
    """
    A2C2f module enhanced with Hybrid TTT & Area-Attention.
    """
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        num_heads = c_ // 32
        # 防止 Head 數量小於 2，導致 Hybrid Attention 拆分錯誤
        if num_heads < 2:
            num_heads = 2

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01  
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(HybridABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + (self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))).to(x.dtype)
        
        return self.cv2(torch.cat(y, 1))
    

# ---------------------------------------------------------
# 修改: A2C2f 整合 TTT 強化
# ---------------------------------------------------------
class TTTC2f(nn.Module):  
    """
    TTTC2f module with residual enhanced feature extraction using TTT/TTTBlock blocks. Also known as R-ELAN
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of TTTlock be a multiple of 32."

        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(TTTBlock(c_, num_heads, mlp_ratio) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer with Attention Residual aggregation."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)

        # 殘差與併發分支整合
        if self.gamma is not None:
            return x + (self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))).to(x.dtype)
        
        return self.cv2(torch.cat(y, 1))