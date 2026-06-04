import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

def autopad(k, p=None, d=1):
    """自動計算 padding 以保證輸出 shape 對齊 YOLO 機制 (與 YOLO11 內部行為一致)"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else[d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class KernelSpatialModulation_Global(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16, 
                 temp=1.0, kernel_temp=None, kernel_att_init='dyconv_as_extra', att_multi=2.0, ksm_only_kernel_att=False, att_grid=1, stride=1, spatial_freq_decompose=False,
                 act_type='sigmoid'):
        super(KernelSpatialModulation_Global, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.ksm_only_kernel_att = ksm_only_kernel_att
        self.kernel_att_init = kernel_att_init
        self.att_multi = att_multi

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.att_grid = att_grid
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()
        self.spatial_freq_decompose = spatial_freq_decompose

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            if spatial_freq_decompose:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes * 2 if self.kernel_size > 1 else in_planes, 1, bias=True)
            else:
                self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or self.ksm_only_kernel_att:  
            self.func_filter = self.skip
        else:
            if spatial_freq_decompose:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes * 2, 1, stride=stride, bias=True)
            else:
                self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or self.ksm_only_kernel_att:  
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
        if hasattr(self, 'spatial_fc') and isinstance(self.spatial_fc, nn.Conv2d):
            nn.init.normal_(self.spatial_fc.weight, std=1e-6)

        if hasattr(self, 'func_filter') and isinstance(self.func_filter, nn.Conv2d):
            nn.init.normal_(self.func_filter.weight, std=1e-6)

        if hasattr(self, 'kernel_fc') and isinstance(self.kernel_fc, nn.Conv2d):
            nn.init.normal_(self.kernel_fc.weight, std=1e-6)
            
        if hasattr(self, 'channel_fc') and isinstance(self.channel_fc, nn.Conv2d):
            nn.init.normal_(self.channel_fc.weight, std=1e-6)

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        if self.act_type =='sigmoid':
            return torch.sigmoid(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi 
        elif self.act_type =='tanh':
            return 1 + torch.tanh_(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)) / self.temperature)
        raise NotImplementedError

    def get_filter_attention(self, x):
        if self.act_type =='sigmoid':
            return torch.sigmoid(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature) * self.att_multi
        elif self.act_type =='tanh':
            return 1 + torch.tanh_(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)) / self.temperature)
        raise NotImplementedError

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size) 
        if self.act_type =='sigmoid':
            return torch.sigmoid(spatial_attention / self.temperature) * self.att_multi
        elif self.act_type =='tanh':
            return 1 + torch.tanh_(spatial_attention / self.temperature)
        raise NotImplementedError

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type =='softmax':
            return F.softmax(kernel_attention / self.kernel_temp, dim=1)
        elif self.act_type =='sigmoid':
            return torch.sigmoid(kernel_attention / self.kernel_temp) * 2 / kernel_attention.size(1)
        elif self.act_type =='tanh':
            return (1 + torch.tanh(kernel_attention / self.kernel_temp)) / kernel_attention.size(1)
        raise NotImplementedError
    
    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint(self._forward, x)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)


class KernelSpatialModulation_Local(nn.Module):
    def __init__(self, channel=None, kernel_num=1, out_n=1, k_size=3, use_global=False):
        super(KernelSpatialModulation_Local, self).__init__()
        self.kn = kernel_num
        self.out_n = out_n
        self.channel = channel
        if channel is not None: k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if self.use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, self.channel // 2 + 1 , 2, dtype=torch.float32) * 1e-6)
        self.norm = nn.LayerNorm(self.channel)

    def forward(self, x, x_std=None):
        orig_dtype = x.dtype  # 紀錄輸入型別
        x = x.squeeze(-1).transpose(-1, -2)
        b, _, c = x.shape
        if self.use_global:
            x_fp32 = x.float()  # 強制轉 FP32 做 FFT
            x_rfft = torch.fft.rfft(x_fp32, dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None].float()
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None].float()
            x_fp32 = x_fp32 + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
            x = x_fp32.to(orig_dtype)  # 運算完畢轉回 FP16/FP32 原型別
        x = self.norm(x)
        att_logit = self.conv(x)
        att_logit = att_logit.reshape(x.size(0), self.kn, self.out_n, c) 
        att_logit = att_logit.permute(0, 1, 3, 2)
        return att_logit
        

def get_fft2freq(d1, d2, use_rfft=False):
    freq_h = torch.fft.fftfreq(d1)
    if use_rfft:
        freq_w = torch.fft.rfftfreq(d2)
    else:
        freq_w = torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing='ij'), dim=-1)
    dist = torch.norm(freq_hw, dim=-1)
    sorted_dist, indices = torch.sort(dist.view(-1))
    
    if use_rfft:
        d2 = d2 // 2 + 1
    sorted_coords = torch.stack([indices // d2, indices % d2], dim=-1)
    return sorted_coords.permute(1, 0), freq_hw


class FrequencyBandModulation(nn.Module):
    def __init__(self, in_channels, k_list=[2], lowfreq_att=False, fs_feat='feat', act='sigmoid', spatial='conv', spatial_group=1, spatial_kernel=3, init='zero', max_size=(64, 64), **kwargs):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.in_channels = in_channels
        self.fs_feat = fs_feat
        self.act = act

        if spatial_group > 64: 
            spatial_group = in_channels
        self.spatial_group = spatial_group

        if spatial == 'conv':
            self.freq_weight_conv_list = nn.ModuleList()
            _n = len(k_list)
            if lowfreq_att:
                _n += 1
            for i in range(_n):
                freq_weight_conv = nn.Conv2d(
                    in_channels=in_channels, out_channels=self.spatial_group, stride=1,
                    kernel_size=spatial_kernel, groups=self.spatial_group, padding=spatial_kernel // 2, bias=True
                )
                if init == 'zero':
                    nn.init.normal_(freq_weight_conv.weight, std=1e-6)
                    if freq_weight_conv.bias is not None:
                        freq_weight_conv.bias.data.zero_()
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError
            
        self.register_buffer('cached_masks', self._precompute_masks(max_size, k_list), persistent=False)

    def _precompute_masks(self, max_size, k_list):
        max_h, max_w = max_size
        _, freq_indices = get_fft2freq(d1=max_h, d2=max_w, use_rfft=True)
        freq_indices = freq_indices.abs().max(dim=-1, keepdims=False)[0]
        masks =[freq_indices < 0.5 / freq + 1e-8 for freq in k_list]
        return torch.stack(masks, dim=0).unsqueeze(1)

    def sp_act(self, freq_weight):
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'tanh':
            return 1 + freq_weight.tanh()
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        raise NotImplementedError

    def forward(self, x, att_feat=None):
        orig_dtype = x.dtype  # 紀錄輸入型別
        if att_feat is None:
            att_feat = x
            
        x_list =[]
        x_fp32 = x.to(torch.float32)  # 強制轉 FP32 穩定做 FFT
        pre_x = x_fp32.clone()
        b, _, h, w = x_fp32.shape
        x_fft = torch.fft.rfft2(x_fp32, norm='ortho')
        
        freq_h, freq_w = h, w // 2 + 1
        current_masks = F.interpolate(self.cached_masks.float(), size=(freq_h, freq_w), mode='nearest')

        for idx, freq in enumerate(self.k_list):
            mask = current_masks[idx]
            low_part = torch.fft.irfft2(x_fft * mask, s=(h, w), norm='ortho')
            high_part = pre_x - low_part
            pre_x = low_part
            
            freq_weight = self.freq_weight_conv_list[idx](att_feat)
            freq_weight = self.sp_act(freq_weight).to(torch.float32)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
            
        if self.lowfreq_att:
            freq_weight = self.freq_weight_conv_list[len(self.k_list)](att_feat)
            freq_weight = self.sp_act(freq_weight).to(torch.float32)
            tmp = freq_weight.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
            
        out = sum(x_list)
        return out.to(orig_dtype)  # 將計算結果轉回輸入型別 (FP16)


class FDConv2d(nn.Conv2d):
    """Core frequency-domain convolution, inherits from nn.Conv2d"""
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 device=None,
                 dtype=None,
                 reduction=0.0625, 
                 kernel_num=4,
                 use_fdconv_if_c_gt=16, 
                 use_fdconv_if_k_in=(1, 3), 
                 use_fdconv_if_stride_in=(1,), 
                 use_fbm_if_k_in=(3,), 
                 use_fbm_for_stride=False,
                 kernel_temp=1.0,
                 temp=None,
                 att_multi=2.0,
                 param_ratio=1,
                 param_reduction=1.0,
                 ksm_only_kernel_att=False,
                 att_grid=1,
                 use_ksm_local=True,
                 ksm_local_act='sigmoid',
                 ksm_global_act='sigmoid',
                 spatial_freq_decompose=False,
                 convert_param=True,
                 linear_mode=False,
                 fbm_cfg=None,
                 **kwargs):
        
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.use_fdconv_if_stride_in = use_fdconv_if_stride_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.spatial_freq_decompose = spatial_freq_decompose
        self.use_fbm_if_k_in = use_fbm_if_k_in

        self.ksm_local_act = ksm_local_act
        self.ksm_global_act = ksm_global_act
        
        if fbm_cfg is None:
            fbm_cfg = {
                'k_list': [2, 4, 8], 'lowfreq_att': False, 'fs_feat': 'feat', 'act': 'sigmoid',
                'spatial': 'conv', 'spatial_group': 1, 'spatial_kernel': 3, 'init': 'zero', 'global_selection': False,
            }

        if self.kernel_num is None:
            self.kernel_num = self.out_channels // 2
            kernel_temp = math.sqrt(self.kernel_num * self.param_ratio)
        if temp is None:
            temp = kernel_temp

        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return

        self.alpha = min(self.out_channels, self.in_channels) // 2 * self.kernel_num * self.param_ratio / param_reduction
        self.KSM_Global = KernelSpatialModulation_Global(self.in_channels, self.out_channels, self.kernel_size[0], groups=self.groups, 
                                                        temp=temp, kernel_temp=kernel_temp,
                                                        reduction=reduction, kernel_num=self.kernel_num * self.param_ratio, 
                                                        kernel_att_init=None, att_multi=att_multi, ksm_only_kernel_att=ksm_only_kernel_att, 
                                                        act_type=self.ksm_global_act, att_grid=att_grid, stride=self.stride, spatial_freq_decompose=spatial_freq_decompose)
        
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, **fbm_cfg)
            
        if self.use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(channel=self.in_channels, kernel_num=1, out_n=int(self.out_channels * self.kernel_size[0] * self.kernel_size[1]) )
        
        self.linear_mode = linear_mode
        self.convert2dftweight(convert_param)
            
    def convert2dftweight(self, convert_param):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)
        weight = self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2).float()
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1))
        
        if self.param_reduction < 1:
            num_to_keep = int(freq_indices.size(1) * self.param_reduction)
            freq_indices = freq_indices[:, :num_to_keep]
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]]
            weight_rfft = weight_rfft.reshape(-1, 2)[None, ].repeat(self.param_ratio, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        else:
            weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2) 
        
        if convert_param:
            self.dft_weight = nn.Parameter(weight_rfft.to(self.weight.dtype), requires_grad=True)
            del self.weight
        else:
            if self.linear_mode:
                self.weight = torch.nn.Parameter(self.weight.squeeze(), requires_grad=True)
                
        indices =[]
        for i in range(self.param_ratio):
            indices.append(freq_indices.reshape(2, self.kernel_num, -1))
        self.register_buffer('indices', torch.stack(indices, dim=0), persistent=False)

    def get_FDW(self):
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        weight = self.weight.reshape(d1, d2, k1, k2).permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2).float()
        weight_rfft = torch.fft.rfft2(weight, dim=(0, 1)).contiguous()
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)[None, ].repeat(self.param_ratio, 1, 1, 1) / (min(self.out_channels, self.in_channels) // 2)
        return weight_rfft.to(self.weight.dtype)
        
    def forward(self, x):
        if min(self.in_channels, self.out_channels) <= self.use_fdconv_if_c_gt or self.kernel_size[0] not in self.use_fdconv_if_k_in:
            return super().forward(x)
            
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        
        if self.use_ksm_local:
            hr_att_logit = self.KSM_Local(global_x)
            hr_att_logit = hr_att_logit.reshape(x.size(0), 1, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1])
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            if self.ksm_local_act == 'sigmoid':
                hr_att = hr_att_logit.sigmoid() * self.att_multi
            elif self.ksm_local_act == 'tanh':
                hr_att = 1 + hr_att_logit.tanh()
        else:
            hr_att = 1
            
        b = x.size(0)
        batch_size, in_planes, height, width = x.size()
        
        # 明確指定 `dtype=torch.float32`，防止 FFT 發生支援度衝突
        DFT_map = torch.zeros((b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2), device=x.device, dtype=torch.float32)
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        dft_weight = self.dft_weight if hasattr(self, 'dft_weight') else self.get_FDW()

        for i in range(self.param_ratio):
            indices = self.indices[i]
            k_att = kernel_attention[:, i].float()  # 強轉 FP32 對齊矩陣
            if self.param_reduction < 1:
                w = dft_weight[i].reshape(self.kernel_num, -1, 2)[None].float()
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * k_att, w[..., 1] * k_att], dim=-1)
            else:
                w = dft_weight[i][indices[0, :, :], indices[1, :, :]][None].float() * self.alpha
                DFT_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack([w[..., 0] * k_att, w[..., 1] * k_att], dim=-1)
                
        # irfft 執行完成後，加上 `.to(x.dtype)` 將資料立刻轉回 FP16 防止下游型別不一致
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(DFT_map), dim=(1, 2)).to(x.dtype)
        adaptive_weights = adaptive_weights.reshape(batch_size, 1, self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1])
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)

        if hasattr(self, 'FBM'):
            x = self.FBM(x)

        if self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1] < (in_planes + self.out_channels) * height * width:
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            aggregate_weight = aggregate_weight.view([-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float): 
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            aggregate_weight = torch.sum(aggregate_weight, dim=1)
            if not isinstance(channel_attention, float): 
                x = x * channel_attention.view(b, -1, 1, 1)
            aggregate_weight = aggregate_weight.view([-1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]])
            x = x.reshape(1, -1, height, width)
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
            if isinstance(filter_attention, float): 
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1))
            else:
                output = output.view(batch_size, self.out_channels, output.size(-2), output.size(-1)) * filter_attention.view(b, -1, 1, 1)
                
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


class FDConv_v2(nn.Module):
    """
    可以直接替換 YOLO11 yaml 中 'Conv' 的封裝模塊
    行為: FDConv2d -> BatchNorm2d -> Activation (預設 SiLU)
    """
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, **kwargs):
        super().__init__()
        # 利用自定義的 FDConv2d 作為核心
        self.conv = FDConv2d(
            in_channels=c1, 
            out_channels=c2, 
            kernel_size=k, 
            stride=s, 
            padding=autopad(k, p, d), 
            groups=g, 
            dilation=d, 
            bias=False,  # YOLO標準中卷積前有 BN，所以不需要 bias
            **kwargs
        )
        self.bn = nn.BatchNorm2d(c2)
        # 支持字串激活函數或布林值
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """用於 YOLO 中 Inference 模式被呼叫融合 BN 的方法"""
        return self.act(self.conv(x))