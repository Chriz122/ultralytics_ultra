import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from timm.models import register_model
from .RepNeXt import ConvNorm, RepDWConvS, RepDWConvM


# ==========================================
# Attention Residuals 所需組件 (論文提出)
# ==========================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    用於歸一化 Key 的幅度，避免歷史層中具有較大輸出的層主導 Softmax。
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x shape: (B, L, C)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SpatialAttnRes(nn.Module):
    """
    Attention Residuals (AttnRes) for spatial feature maps.
    Ref: Technical Report of Attention Residuals (Kimi Team)
    透過自適應權重動態聚合 Depth 方向的特徵，取代傳統硬加總的 Shortcut。
    """
    def __init__(self, dim):
        super().__init__()
        # 論文規定: "Crucially, all pseudo-query vectors must be initialized to zero."
        # 這確保了初始化時對歷史層具有相同的注意力權重 (均勻分佈)，避免訓練初期的不穩定。
        self.w = nn.Parameter(torch.zeros(dim))
        self.norm = RMSNorm(dim, eps=1e-6) 

    def forward(self, past_states):
        """
        past_states: 歷史層特徵列表 List[Tensor], Tensor shape=(B, C, H, W)
        """
        # 如果只有一層歷史狀態，直接返回 (無需計算 attention)
        if len(past_states) == 1:
            return past_states[0]

        # 1. 取得 Keys: 透過 Global Average Pooling 將 (B, C, H, W) 壓縮為 (B, C)
        keys = [F.adaptive_avg_pool2d(v, 1).flatten(1) for v in past_states]
        keys = torch.stack(keys, dim=1)  # shape: (B, L, C)，L為歷史層數

        # 2. Key 進行 RMSNorm 正規化 (防止個別層幅度爆炸)
        keys = self.norm(keys)           # shape: (B, L, C)

        # 3. 計算 Attention Logits (Pseudo-query 點乘 Keys)
        # w: (C,), keys: (B, L, C) -> logits: (B, L)
        logits = torch.einsum('c, b l c -> b l', self.w, keys)

        # 4. Softmax over depth (跨層的注意力權重分配)
        attn = F.softmax(logits, dim=-1) # shape: (B, L)

        # 5. Aggregate Values (加權歷史層特徵)
        # 使用 unbind 和 view 避免對 4D tensors 進行高成本的 stack，優化顯存佔用
        out = sum(a.view(-1, 1, 1, 1) * v for a, v in zip(attn.unbind(1), past_states))
        
        return out.to(past_states[0].dtype)


# ==========================================
# 原有網路組件區 (基礎 Conv, Norm 等)
# ==========================================
# class ConvNorm(nn.Sequential):
#     def __init__(
#         self, in_channels, out_channels, kernel_size=1, stride=1,
#         padding=0, dilation=1, groups=1, bias=False, bn_weight_init=1,
#     ):
#         super().__init__()
#         self.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias))
#         self.add_module("norm", nn.BatchNorm2d(out_channels))
#         nn.init.constant_(self.norm.weight, bn_weight_init)
#         nn.init.constant_(self.norm.bias, 0)

#     @torch.no_grad()
#     def fuse(self):
#         w = self.norm.weight / (self.norm.running_var + self.norm.eps) ** 0.5
#         b = self.norm.bias - w * self.norm.running_mean

#         if self.conv.bias is not None:
#             b += w * self.conv.bias

#         w = w[:, None, None, None] * self.conv.weight

#         m = nn.Conv2d(
#             w.size(1) * self.conv.groups, w.size(0), w.shape[2:],
#             stride=self.conv.stride, padding=self.conv.padding,
#             dilation=self.conv.dilation, groups=self.conv.groups, device=self.conv.weight.device,
#         )
#         m.weight.data.copy_(w)
#         m.bias.data.copy_(b)
#         return m


class NormLinear(nn.Sequential):
    def __init__(self, in_channels, out_channels, bias=True, std=0.02):
        super().__init__()
        self.add_module("norm", nn.BatchNorm1d(in_channels))
        self.add_module("linear", nn.Linear(in_channels, out_channels, bias=bias))
        trunc_normal_(self.linear.weight, std=std)
        if bias:
            nn.init.constant_(self.linear.bias, 0)

    @torch.no_grad()
    def fuse(self):
        norm, linear = self._modules.values()
        w = norm.weight / (norm.running_var + norm.eps) ** 0.5
        b = norm.bias - self.norm.running_mean * self.norm.weight / (norm.running_var + norm.eps) ** 0.5
        w = linear.weight * w[None, :]
        if linear.bias is None:
            b = b @ self.linear.weight.T
        else:
            b = (linear.weight @ b[:, None]).view(-1) + self.linear.bias
        m = nn.Linear(w.size(1), w.size(0), device=linear.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def mlp(in_channels, hidden_channels, act_layer=nn.GELU):
    return nn.Sequential(
        ConvNorm(in_channels, hidden_channels, kernel_size=1),
        act_layer(),
        ConvNorm(hidden_channels, in_channels, kernel_size=1),
    )


# class RepDWConvS(nn.Module):
#     def __init__(self, in_channels, stride=1, bias=True):
#         super().__init__()
#         self.stride = stride
#         kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
#         self.conv_3_3 = nn.Conv2d(bias=bias, kernel_size=3, stride=stride, dilation=1, padding=1, **kwargs)
#         self.conv_3_w = nn.Conv2d(bias=bias and stride==1, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), **kwargs)
#         self.conv_3_h = nn.Conv2d(bias=bias and stride==1, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), **kwargs)
#         self.conv_2_2 = nn.Conv2d(bias=bias, kernel_size=2, stride=stride, dilation=2, padding=1, **kwargs)

#     def forward(self, x):
#         if self.stride == 1:
#             return self.conv_3_3(x) + self.conv_3_h(x) + self.conv_3_w(x) + self.conv_2_2(x)
#         return self.conv_3_3(x) + self.conv_3_h(self.conv_3_w(x)) + self.conv_2_2(x)

#     @torch.no_grad()
#     def fuse(self):
#         # 融合邏輯保留
#         conv_3_3_w, conv_3_3_b = self.conv_3_3.weight, self.conv_3_3.bias
#         conv_2_2_w, conv_2_2_b = self.conv_2_2.weight, self.conv_2_2.bias
#         conv_3_w_w, conv_3_w_b = self.conv_3_w.weight, self.conv_3_w.bias
#         conv_3_h_w, conv_3_h_b = self.conv_3_h.weight, self.conv_3_h.bias

#         conv_2_2_w = nn.functional.conv_transpose2d(conv_2_2_w, torch.ones((1, 1, 1, 1), device=conv_2_2_w.device), stride=2)
#         if self.stride == 2:
#             conv_stack_3_w = torch.einsum("bcnx,bcyn->bcyx", conv_3_w_w, conv_3_h_w)
#             w = conv_3_3_w + conv_stack_3_w + conv_2_2_w
#         else:
#             conv_3_w_w = nn.functional.pad(conv_3_w_w, [0, 0, 1, 1])
#             conv_3_h_w = nn.functional.pad(conv_3_h_w, [1, 1, 0, 0])
#             w = conv_3_3_w + conv_3_w_w + conv_3_h_w + conv_2_2_w
#         self.conv_3_3.weight.data.copy_(w)

#         if conv_3_3_b is not None:
#             b = conv_3_3_b + conv_2_2_b
#             if self.stride == 1:
#                 b += conv_3_w_b + conv_3_h_b
#             self.conv_3_3.bias.data.copy_(b)
#         return self.conv_3_3


# class RepDWConvM(nn.Module):
#     def __init__(self, in_channels, stride=1, bias=True):
#         super().__init__()
#         kwargs = {"in_channels": in_channels, "out_channels": in_channels, "groups": in_channels}
#         self.conv_7_7 = nn.Conv2d(bias=bias, kernel_size=(7, 7), stride=stride, padding=3, **kwargs)
#         self.conv_5_3 = nn.Conv2d(bias=bias, kernel_size=(5, 3), stride=stride, padding=(2, 1), **kwargs)
#         self.conv_3_5 = nn.Conv2d(bias=bias, kernel_size=(3, 5), stride=stride, padding=(1, 2), **kwargs)
#         self.conv_7_w = nn.Conv2d(bias=False, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), **kwargs)
#         self.conv_7_h = nn.Conv2d(bias=False, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), **kwargs)
#         self.conv_5_w = nn.Conv2d(bias=False, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2), **kwargs)
#         self.conv_5_h = nn.Conv2d(bias=False, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0), **kwargs)

#     def forward(self, x):
#         return self.conv_7_7(x) + self.conv_5_3(x) + self.conv_3_5(x) + self.conv_7_h(self.conv_7_w(x)) + self.conv_5_h(self.conv_5_w(x))

#     @torch.no_grad()
#     def fuse(self):
#         # 融合邏輯保留
#         conv_7_7_w, conv_7_7_b = self.conv_7_7.weight, self.conv_7_7.bias
#         conv_5_3_w, conv_5_3_b = self.conv_5_3.weight, self.conv_5_3.bias
#         conv_3_5_w, conv_3_5_b = self.conv_3_5.weight, self.conv_3_5.bias
#         conv_7_w_w, conv_7_h_w = self.conv_7_w.weight, self.conv_7_h.weight
#         conv_5_w_w, conv_5_h_w = self.conv_5_w.weight, self.conv_5_h.weight

#         conv_5_3_w = nn.functional.pad(conv_5_3_w, [2, 2, 1, 1])
#         conv_3_5_w = nn.functional.pad(conv_3_5_w, [1, 1, 2, 2])

#         conv_stack_7_w = torch.einsum("bcnx,bcyn->bcyx", conv_7_w_w, conv_7_h_w)
#         conv_stack_5_w = torch.einsum("bcnx,bcyn->bcyx", conv_5_w_w, conv_5_h_w)
#         conv_stack_5_w = nn.functional.pad(conv_stack_5_w, [1, 1, 1, 1])

#         w = conv_7_7_w + conv_5_3_w + conv_3_5_w + conv_stack_7_w + conv_stack_5_w
#         self.conv_7_7.weight.data.copy_(w)

#         if conv_7_7_b is not None:
#             b = conv_7_7_b + conv_5_3_b + conv_3_5_b
#             self.conv_7_7.bias.data.copy_(b)
#         return self.conv_7_7


class ChunkConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        assert in_channels % 4 == 0
        hidden_channels = in_channels // 4
        self.conv_s = RepDWConvS(hidden_channels)
        self.conv_m = RepDWConvM(hidden_channels)
        self.conv_l = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(1, 11), padding=(0, 5), groups=hidden_channels),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(11, 1), padding=(5, 0), groups=hidden_channels),
        )

    def forward(self, x):
        i, s, m, l = torch.chunk(x, 4, dim=1)
        return torch.cat((i, self.conv_s(s), self.conv_m(m), self.conv_l(l)), dim=1)


class CopyConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_s = RepDWConvS(in_channels, stride=2)
        self.conv_m = RepDWConvM(in_channels, stride=2)

    def forward(self, x):
        return torch.cat((self.conv_s(x), self.conv_m(x)), dim=1)


class RepNextStem(nn.Module):
    def __init__(self, in_channels, out_channels, act_layer=nn.GELU, kernel_size=3, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        kwargs = {"kernel_size": kernel_size, "stride": stride, "padding": padding}
        self.stem = nn.Sequential(
            ConvNorm(in_channels, out_channels // 2, **kwargs),
            act_layer(),
            ConvNorm(out_channels // 2, out_channels, **kwargs),
        )

    def forward(self, x):
        return self.stem(x)


# ==========================================
# 加入 Attention Residuals 理念的模塊
# ==========================================

class MetaNeXtBlock(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU):
        super().__init__()
        self.token_mixer = ChunkConv(in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.channel_mixer = mlp(in_channels, in_channels * mlp_ratio, act_layer=act_layer)

    def forward(self, x):
        # 【重要修改】移除原先的 `x + ...`
        # 根據 AttnRes 理論，本模塊只負責做特徵變換 (Transformation) 也就是 f_l(h_l)
        # 跨層聚合交由外部的 RepNextStage 使用 SpatialAttnRes 管理
        return self.channel_mixer(self.norm(self.token_mixer(x)))


class Downsample(nn.Module):
    def __init__(self, in_channels, mlp_ratio, act_layer=nn.GELU):
        super().__init__()
        out_channels = in_channels * 2
        self.token_mixer = CopyConv(in_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.channel_mixer = mlp(out_channels, out_channels * mlp_ratio, act_layer=act_layer)

    def forward(self, x):
        # 降採樣層由於維度變化，暫時保留標準的跨步殘差設計
        x = self.norm(self.token_mixer(x))
        return x + self.channel_mixer(x)


class RepNextStage(nn.Module):
    """
    結合 Attention Residuals 機制的階段 (Stage)
    將傳統依序相加的層，轉換為收集 past_states，並使用 AttnRes 進行聚合運算。
    """
    def __init__(self, in_channels, out_channels, depth, mlp_ratio, act_layer=nn.GELU, downsample=True):
        super().__init__()
        self.downsample = Downsample(in_channels, mlp_ratio, act_layer=act_layer) if downsample else nn.Identity()
        
        # 定義深度相同的特徵變換 Blocks
        self.blocks = nn.ModuleList([
            MetaNeXtBlock(out_channels, mlp_ratio, act_layer=act_layer) 
            for _ in range(depth)
        ])
        
        # 每個 Block 的輸入之前都需要聚合歷史狀態，且 Stage 最終輸出也需要一次聚合
        # 所以總共需要 depth + 1 個 SpatialAttnRes
        self.attn_res_modules = nn.ModuleList([
            SpatialAttnRes(out_channels) for _ in range(depth + 1)
        ])

    def forward(self, x):
        x = self.downsample(x)
        
        # past_states 儲存過往所有層的結果 (包含初始輸入)
        past_states = [x]
        
        for i, block in enumerate(self.blocks):
            # 1. 將過往所有狀態透過 AttnRes 動態聚合，得出本層的輸入 h_l
            h_l = self.attn_res_modules[i](past_states)
            
            # 2. 將 h_l 送入區塊進行特徵變換，得出本層新產生的輸出 v_l
            v_l = block(h_l)
            
            # 3. 將本層新特徵加入歷史池
            past_states.append(v_l)
        
        # 取最後一個 AttnRes 來融合所有狀態，產生本階段的最終輸出
        out = self.attn_res_modules[-1](past_states)
        return out


class RepNextClassifier(nn.Module):
    def __init__(self, dim, num_classes, distillation=False, drop=0.0):
        super().__init__()
        self.head_drop = nn.Dropout(drop)
        self.head = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.distillation = distillation
        self.num_classes = num_classes
        self.head_dist = NormLinear(dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.head_drop(x)
        x1, x2 = self.head(x), self.head_dist(x)
        if self.training and self.distillation and not torch.jit.is_scripting():
            return x1, x2
        else:
            return (x1 + x2) / 2

    @torch.no_grad()
    def fuse(self):
        if not self.num_classes > 0:
            return nn.Identity()
        head = self.head.fuse()
        head_dist = self.head_dist.fuse()
        head.weight += head_dist.weight
        head.bias += head_dist.bias
        head.weight /= 2
        head.bias /= 2
        return head


class RepNext(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=(48, 96, 192, 384),
        depth=(2, 2, 9, 1),
        mlp_ratio=2,
        global_pool="avg",
        num_classes=1000,
        act_layer=nn.GELU,
        distillation=False,
        drop_rate=0.0,
        img_size=224,
        **kwargs,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_size = img_size

        in_channels = embed_dim[0]
        self.stem = RepNextStem(in_chans, in_channels, act_layer=act_layer)
        stride = 4
        self.feature_info = []
        stages = []
        
        for i in range(len(embed_dim)):
            downsample = True if i != 0 else False
            stages.append(
                RepNextStage(
                    in_channels, embed_dim[i], depth[i],
                    mlp_ratio=mlp_ratio, act_layer=act_layer, downsample=downsample,
                )
            )
            stage_stride = 2 if downsample else 1
            stride *= stage_stride
            self.feature_info += [dict(num_chs=embed_dim[i], reduction=stride, module=f"stages.{i}")]
            in_channels = embed_dim[i]
        
        self.stages = nn.Sequential(*stages)

        self.num_features = embed_dim[-1]
        self.head_drop = nn.Dropout(drop_rate)
        self.head = RepNextClassifier(embed_dim[-1], num_classes, distillation)

        # width_list dummy forward 機制
        self.width_list = []
        try:
            self.eval()
            dummy_input = torch.randn(1, in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                 features = self.forward(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train()
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            self.width_list = list(self.embed_dim)
            self.train()

    def forward_features(self, x):
        features = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

    def forward_head(self, x):
        if self.global_pool == "avg":
            x = x.mean((2, 3), keepdim=False)
        return self.head(self.head_drop(x))

    def forward(self, x):
        features = self.forward_features(x)
        return features

    @torch.no_grad()
    def fuse(self):
        def fuse_children(net):
            for child_name, child in net.named_children():
                if hasattr(child, "fuse"):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        fuse_children(self)


# ==========================================
# 註冊模型系列
# ==========================================

@register_model
def repnext_attnres_m0(pretrained=False, img_size=224, **kwargs):
    return RepNext(embed_dim=(40, 80, 160, 320), depth=(2, 2, 9, 1), img_size=img_size, **kwargs)

@register_model
def repnext_attnres_m1(pretrained=False, img_size=224, **kwargs):
    return RepNext(embed_dim=(48, 96, 192, 384), depth=(3, 3, 15, 2), img_size=img_size, **kwargs)

@register_model
def repnext_attnres_m2(pretrained=False, img_size=224, **kwargs):
    return RepNext(embed_dim=(56, 112, 224, 448), depth=(3, 3, 15, 2), img_size=img_size, **kwargs)

@register_model
def repnext_attnres_m3(pretrained=False, img_size=224, **kwargs):
    return RepNext(embed_dim=(64, 128, 256, 512), depth=(3, 3, 13, 2), img_size=img_size, **kwargs)

@register_model
def repnext_attnres_m4(pretrained=False, img_size=224, **kwargs):
    return RepNext(embed_dim=(64, 128, 256, 512), depth=(5, 5, 25, 4), img_size=img_size, **kwargs)

@register_model
def repnext_attnres_m5(pretrained=False, img_size=224, **kwargs):
    return RepNext(embed_dim=(80, 160, 320, 640), depth=(7, 7, 35, 2), img_size=img_size, **kwargs)


# 測試入口
if __name__ == "__main__":
    img_h, img_w = 640, 640
    print("--- Creating RepNext M3 with Attention Residuals ---")
    
    model = repnext_m3(img_size=img_h)
    model.eval()
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    input_tensor = torch.randn(1, 3, img_h, img_w)
    print(f"\n--- Testing forward pass (Input: {input_tensor.shape}) ---")
    
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

        try:
            import pytorch_model_summary
            print(pytorch_model_summary.summary(model, input_tensor))
        except ModuleNotFoundError:
            print("\npytorch_model_summary 不存在，已略過摘要印出")

    except Exception as e:
        print(f"\nError during testing: {e}")