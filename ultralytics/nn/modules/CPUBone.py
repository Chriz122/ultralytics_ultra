# CPUBone: Efficient Vision Backbone Design for Devices with Low Parallelization Capabilities
# Adapted for YOLO integration similar to SMT Backbones

import os
import math
from inspect import signature
from copy import deepcopy
import torch

import torch.nn.functional as F
import torch.nn as nn
from functools import partial


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)
    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def build_kwargs_from_config(config: dict, target_func: callable) -> dict[str, any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs


def load_state_dict_from_file(file: str, only_state_dict=True) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(file, map_location="cpu")
    if "epoch" in checkpoint:
        print("checkpoint from epoch %d and its best validation result is %.3f" % (checkpoint["epoch"],checkpoint["best_val"]))
    if only_state_dict and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def remap_legacy_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for k, v in state_dict.items():
        k = k.replace(".conv_proj.0.", ".conv_proj.conv.")
        k = k.replace(".conv_proj.1.", ".conv_proj.norm.")
        remapped[k] = v
    return remapped


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    REGISTERED_NORM_DICT: dict[str, type] = {
        "bn2d": nn.BatchNorm2d,
        "ln": nn.LayerNorm,
    }
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def build_act(name: str, **kwargs) -> nn.Module or None:
    REGISTERED_ACT_DICT: dict[str, type] = {
        "relu": nn.ReLU,
        "relu6": nn.ReLU6,
        "hswish": nn.Hardswish,
        "silu": nn.SiLU,
        "gelu": partial(nn.GELU, approximate="tanh"),
    }
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None


def get_same_padding(kernel_size: int or tuple[int, ...], stride=1) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    elif kernel_size == 2:
        if stride==2:
            return 0
        return -1
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


#########################################    MODEL MODULES     ######################################################

class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bias=True, dropout=0, norm=None, act_func=None, squeeze_it=False):
        super(LinearLayer, self).__init__()
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)
        self.squeeze_it = squeeze_it

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if self.squeeze_it:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if not self.dropout is None:
            x = self.dropout(x)
        x = self.linear(x)
        if not self.norm is None:
            x = self.norm(x)
        if not self.act is None:
            x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, main: nn.Module or None, shortcut: nn.Module or None, post_act=None, pre_norm: nn.Module or None = None):
        super(ResidualBlock, self).__init__()
        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if not self.post_act is None:
                res = self.post_act(res)
        return res


class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, dilation=1, groups=1, use_bias=False, dropout=0, norm="bn2d", act_func="relu"):
        super(ConvLayer, self).__init__()
        padding = get_same_padding(kernel_size, stride)
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        
        if padding == -1:
            self.conv = nn.Sequential(
                torch.nn.ZeroPad2d((1,0,1,0)),
                nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=0, dilation=(dilation, dilation), groups=groups, bias=use_bias)
            )
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=padding, dilation=(dilation, dilation), groups=groups, bias=use_bias)
            
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if not self.norm is None:
            x = self.norm(x)
        if not self.act is None:
            x = self.act(x)
        return x


class MBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, mid_channels=None, expand_ratio=6, grouping=1, use_bias=False, norm=("bn2d", "bn2d", "bn2d"), act_func=("relu6", "relu6", None)):
        super(MBConv, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(in_channels, mid_channels, 1, stride=1, groups=grouping, norm=norm[0], act_func=act_func[0], use_bias=use_bias[0])
        self.depth_conv = ConvLayer(mid_channels, mid_channels, kernel_size, stride=stride, groups=mid_channels, norm=norm[1], act_func=act_func[1], use_bias=use_bias[1])
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, groups=1, norm=norm[2], act_func=act_func[2], use_bias=use_bias[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x
    

class FusedMBConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, mid_channels=None, expand_ratio=6, groups=1, use_bias=False, norm=("bn2d", "bn2d"), act_func=("relu6", None)):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(in_channels, mid_channels, kernel_size, stride, groups=groups, use_bias=use_bias[0], norm=norm[0], act_func=act_func[0])
        self.point_conv = ConvLayer(mid_channels, out_channels, 1, groups=1, use_bias=use_bias[1], norm=norm[1], act_func=act_func[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class SDALayer(nn.Module):
    def scaled_dot_product(self, q, k, v):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values

    def forward(self, q, k, v) -> torch.Tensor:
        return self.scaled_dot_product(q, k, v)


class ConvAttention(nn.Module):
    def __init__(self, input_dim, head_dim_mul=1.0, att_stride=4, att_kernel=7, fuseconv=False, smallkernel=False, lose_transpose=False):
        super().__init__()
        self.head_dim_mul = head_dim_mul
        self.num_heads = int(max(1, (input_dim * self.head_dim_mul) // 30))
        self.input_dim = input_dim
        self.head_dim = int((input_dim // self.num_heads) * self.head_dim_mul)
        self.num_keys = 3
        self.att_stride = att_stride

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvLayer(input_dim, input_dim, kernel_size=2 if smallkernel else att_kernel, norm="bn2d", act_func=None, stride=att_stride, groups=input_dim)
        self.pwise = nn.Sequential(nn.Conv2d(input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias=False))
        self.sda = SDALayer()

        self.o_proj_inpdim = self.head_dim * self.num_heads
        self.o_proj = nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)

        self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=att_stride*2, stride=att_stride, padding=att_stride//2, groups=input_dim)
        if att_stride == 1:
            self.upsampling = nn.ConvTranspose2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1, groups=input_dim)

        if fuseconv:
            self.o_proj = nn.Identity()
            if att_stride == 1:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=3, stride=1, padding=1)
            else:
                self.upsampling = nn.ConvTranspose2d(self.o_proj_inpdim, input_dim, kernel_size=att_stride*2, stride=att_stride, padding=att_stride//2)

        if lose_transpose:
            upsampling = [nn.Upsample(scale_factor=att_stride, mode="nearest") if att_stride > 1 else nn.Identity()]
            if fuseconv:
                upsampling = [nn.Conv2d(self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0)] + upsampling
            self.upsampling = nn.Sequential(*upsampling)

    def forward(self, x):
        N, C, H, W = x.size()
        xout = self.conv_proj(x)
        xout = self.pwise(xout)

        N, c, h, w = xout.size()
        qkv = xout.reshape(N, self.num_heads, self.num_keys * self.head_dim, h * w)
        qkv = qkv.permute(0, 1, 3, 2)  # [N, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=3)

        values = self.sda(q, k, v)
        o = self.o_proj(values.permute(0, 1, 3, 2).reshape(N, self.o_proj_inpdim, h, w))

        o = self.upsampling(o)
        return o[:N, :C, :H, :W]


class CPUBoneBlock(nn.Module):
    def __init__(self, in_channels: int, expand_ratio: float = 4, norm="bn2d", act_func="hswish", fuseconv=False, bb_convattention=False, bb_convin2=False, grouping=1, att_stride=1, mlpexpans=4, smallkernel=False, lose_transpose=False):
        super(CPUBoneBlock, self).__init__()
        att_kernel = 5 if att_stride > 1 else 3

        block = ConvAttention(input_dim=in_channels, att_stride=att_stride, att_kernel=att_kernel, head_dim_mul=0.5, fuseconv=fuseconv, smallkernel=smallkernel, lose_transpose=lose_transpose)

        context_module = ResidualBlock(nn.Sequential(nn.GroupNorm(1, in_channels), block), IdentityLayer())
        mlp = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, in_channels * mlpexpans, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels * mlpexpans, in_channels, kernel_size=1),
            nn.Dropout(p=0.1),
        )
        context_module = nn.Sequential(context_module, ResidualBlock(mlp, IdentityLayer()))

        if fuseconv and in_channels < 256:
            local_module = FusedMBConv(in_channels=in_channels, out_channels=in_channels, expand_ratio=expand_ratio, use_bias=(True, False), kernel_size=2 if smallkernel else 3, groups=grouping, norm=norm, act_func=(act_func, None))
        else:
            local_module = MBConv(in_channels=in_channels, out_channels=in_channels, expand_ratio=expand_ratio, grouping=grouping, use_bias=(True, True, False), kernel_size=2 if smallkernel else 3, norm=(None, None, norm), act_func=(act_func, act_func, None))

        self.total = nn.Sequential(context_module, ResidualBlock(local_module, IdentityLayer()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.total(x)


##############################    CPUBone Backbone Architecture      ###############################

class CPUBoneBackbone(nn.Module):
    def __init__(self, width_list: list[int], depth_list: list[int], in_channels=3, img_size=224, expand_ratio=4, norm="bn2d", act_func="hswish", bb_convattention=False, bb_convin2=False, fastit=False, huge_model=False, bigit=False, grouping=1, smallk_only_lasts=False, lose_transpose=False, just_unfused=False) -> None:
        super().__init__()

        self.in_chans = in_channels
        self.img_size = img_size
        
        fuseconv = fastit and not just_unfused
        stage_num = 0

        ### STAGE 0 (Stem)
        self.input_stem = [
            ConvLayer(in_channels=in_channels, out_channels=width_list[0], kernel_size=3, stride=2, norm=norm, act_func=act_func)
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(in_channels=width_list[0], out_channels=width_list[0], stride=1, expand_ratio=4 if huge_model else 2, fusedmbconv=fuseconv, grouping=grouping, norm=norm, act_func=act_func)
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))

        curr_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        stage_num += 1

        ### STAGE 1-2
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(in_channels=curr_channels, out_channels=w, stride=stride, expand_ratio=6 if stride == 2 and (bigit or huge_model) else expand_ratio, fusedmbconv=fuseconv, grouping=grouping, norm=norm, act_func=act_func)
                stage.append(ResidualBlock(block, IdentityLayer() if stride == 1 else None))
                curr_channels = w
            self.stages.append(OpSequential(stage))
            stage_num += 1

        ### STAGE 3-4
        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(in_channels=curr_channels, out_channels=w, stride=2, expand_ratio=6 if bigit or (huge_model and stage_num < 4) else expand_ratio, fusedmbconv=fastit and (not huge_model or stage_num < 5), grouping=grouping, norm=norm, act_func=act_func)
            stage.append(ResidualBlock(block, None))
            curr_channels = w

            for _ in range(d):
                stage.append(
                    CPUBoneBlock(in_channels=curr_channels, expand_ratio=expand_ratio, norm=norm, act_func=act_func, bb_convattention=bb_convattention, fuseconv=fuseconv, bb_convin2=bb_convin2, grouping=grouping, att_stride=2 if stage_num == 3 else 1, mlpexpans=4 if fastit else 2, smallkernel=smallk_only_lasts, lose_transpose=lose_transpose)
                )

            self.stages.append(OpSequential(stage))
            stage_num += 1

        self.stages = nn.ModuleList(self.stages)

        # --- Dynamic width_list calculation (Identical to SMT) ---
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
            print("Setting width_list as fallback (Stage 1 to 4).")
            self.width_list = width_list[1:]
            self.train()

    @staticmethod
    def build_local_block(in_channels: int, out_channels: int, stride: int, expand_ratio: float, norm: str, act_func: str, fusedmbconv: bool = False, grouping: int = 1, kernel_size: int = 3) -> nn.Module:
        if fusedmbconv:
            return FusedMBConv(in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, use_bias=False, kernel_size=kernel_size, groups=grouping, norm=norm, act_func=(act_func, None))
        else:
            return MBConv(in_channels=in_channels, out_channels=out_channels, stride=stride, expand_ratio=expand_ratio, kernel_size=kernel_size, grouping=grouping, use_bias=False, norm=(None, None, norm), act_func=(act_func, act_func, None))

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        # 通過 Stage 0 (Stem)，但不將其加入到 YOLO 需要的特徵列表中
        x = self.input_stem(x)
        
        # 依序通過 Stage 1 到 Stage 4
        for stage in self.stages:
            x = stage(x)
            features.append(x)  # 返回 List 而非 dict，避免 YOLO 的 insert AttributeError

        return features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.forward_features(x)


# ====================================================================================
# Model Factory Functions (Modified to match SMT style usage)
# ====================================================================================

def _create_cpubone(width_list, depth_list, pretrained=False, checkpoint_path=None, **kwargs):
    model = CPUBoneBackbone(
        width_list=width_list,
        depth_list=depth_list,
        **build_kwargs_from_config(kwargs, CPUBoneBackbone)
    )
    if pretrained and checkpoint_path and os.path.exists(checkpoint_path):
        try:
            weight = load_state_dict_from_file(checkpoint_path)
            weight = remap_legacy_state_dict(weight)
            # 因為我們只返回 backbone，所以剔除含有 cls head 的權重
            weight = {k.replace('backbone.', ''): v for k, v in weight.items() if k.startswith('backbone.')}
            model.load_state_dict(weight, strict=False)
            print("Pretrained weights loaded successfully.")
        except Exception as e:
            print("Model weights could not be loaded:", e)
            
    return model


def cpubone_nano(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 1, 2], pretrained=pretrained, **kwargs)

def cpubone_t0(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 1, 3], pretrained=pretrained, **kwargs)

def cpubone_s0(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 2, 3], pretrained=pretrained, **kwargs)

def cpubone_s1(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[14, 28, 56, 112, 224], depth_list=[0, 1, 1, 2, 3], pretrained=pretrained, **kwargs)

def cpubone_b0(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 3, 4], pretrained=pretrained, **kwargs)

def cpubone_b1(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[16, 32, 64, 128, 256], depth_list=[0, 1, 1, 5, 5], pretrained=pretrained, **kwargs)

def cpubone_b15(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[20, 40, 80, 160, 320], depth_list=[0, 1, 1, 6, 6], pretrained=pretrained, **kwargs)

def cpubone_b2(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[24, 48, 96, 192, 384], depth_list=[0, 1, 1, 6, 6], pretrained=pretrained, **kwargs)

def cpubone_b25(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[24, 48, 96, 192, 384], depth_list=[0, 2, 3, 6, 6], pretrained=pretrained, **kwargs)

def cpubone_b3(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[32, 64, 128, 256, 512], depth_list=[1, 2, 3, 6, 6], pretrained=pretrained, **kwargs)

def cpubone_b4(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[64, 128, 256, 512, 1024], depth_list=[2, 3, 6, 12, 8], pretrained=pretrained, **kwargs)

def cpubone_b5(pretrained=False, **kwargs):
    return _create_cpubone(width_list=[128, 256, 512, 1024, 2048], depth_list=[2, 4, 5, 20, 10], pretrained=pretrained, **kwargs)


if __name__ == '__main__':
    img_h, img_w = 224, 224
    print("--- Creating CPUBone B1 model ---")
    model = cpubone_b1()
    print("Model created successfully.")
    print("Calculated width_list:", model.width_list)

    # Test forward pass
    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- Testing CPUBone B1 forward pass (Input: {input_tensor.shape}) ---")

    model.eval()
    try:
        with torch.no_grad():
            output_features = model(input_tensor)
        print("Forward pass successful.")
        
        # Checking Outputs List Format (To avoid YOLO AttributeError)
        assert isinstance(output_features, list), "Output must be a list for YOLO integration!"
        print("Output feature shapes (Should only be Stage 1 ~ Stage 4):")
        for i, features in enumerate(output_features):
            print(f"Stage {i+1}: {features.shape}")

        # Verify width_list matches runtime output
        runtime_widths = [f.size(1) for f in output_features]
        print("\nRuntime output feature channels:", runtime_widths)
        assert model.width_list == runtime_widths, "Width list mismatch!"
        print("Width list verified successfully. Ready for YOLO integration.")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()