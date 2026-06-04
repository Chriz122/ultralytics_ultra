import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings

# ================= 移除 mmcv，純 PyTorch 實現 =================

def xavier_init(module: nn.Module, gain: float = 1, bias: float = 0, distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def carafe(x, normed_mask, kernel_size, group=1, up=1):
    """純 PyTorch 版 CARAFE，針對 YOLO 尺寸對齊做了優化"""
    b, c, h, w = x.shape
    _, m_c, m_h, m_w = normed_mask.shape
    pad = kernel_size // 2
    pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
    unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
    unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
    
    # YOLO 優化：使用確切的目標 size 進行插值，避免奇數分辨率造成的維度崩潰
    unfold_x = F.interpolate(unfold_x, size=(m_h, m_w), mode='nearest')
    
    unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
    normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
    res = unfold_x * normed_mask
    res = res.sum(dim=2).reshape(b, c, m_h, m_w)
    return res

# ================= 移除 numpy，純 PyTorch 實現 =================

def hamming2D(M, N):
    """生成二維 Hamming 窗 (純 PyTorch 實現)"""
    hamming_x = torch.hamming_window(M, periodic=False)
    hamming_y = torch.hamming_window(N, periodic=False)
    hamming_2d = torch.outer(hamming_x, hamming_y)
    return hamming_2d

# ================= FreqFusion 核心模組 =================

class FreqFusion(nn.Module):
    def __init__(self,
                 c1, # 接收 YOLO 傳來的通道列表 [lr_channels, hr_channels]
                 c2=None, # YOLO 自動計算輸出通道時的預留位
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,        
                 align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=False,
                 feature_resample_group=4,
                 comp_feat_upsample=True,
                 use_high_pass=True,
                 use_low_pass=True,
                 hr_residual=True,
                 semi_conv=True,
                 hamming_window=True,
                 feature_resample_norm=True,
                 **kwargs):
        super().__init__()
        # YOLO 兼容性：強制要求輸入為 [低分辨率特徵, 高分辨率特徵]
        assert isinstance(c1, list) and len(c1) == 2, "FreqFusion in YOLO requires c1 to be a list [lr_channels, hr_channels]"
        lr_channels, hr_channels = c1

        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        
        self.align_corners = align_corners
        self.upsample_mode = upsample_mode
        self.hr_residual = hr_residual
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.semi_conv = semi_conv
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        
        if self.feature_resample:
            self.dysampler = LocalSimGuidedSampler(in_channels=compressed_channels, scale=2, style='lp', groups=feature_resample_group, use_direct_scale=True, kernel_size=encoder_kernel, norm=feature_resample_norm)
        if self.use_high_pass:
            self.content_encoder2 = nn.Conv2d(
                self.compressed_channels,
                highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                self.encoder_kernel,
                padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                dilation=self.encoder_dilation,
                groups=1)
                
        self.hamming_window = hamming_window
        lowpass_pad = 0
        highpass_pad = 0
        
        if self.hamming_window:
            self.register_buffer('hamming_lowpass', hamming2D(lowpass_kernel + 2 * lowpass_pad, lowpass_kernel + 2 * lowpass_pad)[None, None,])
            self.register_buffer('hamming_highpass', hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad)[None, None,])
        else:
            self.register_buffer('hamming_lowpass', torch.FloatTensor([1.0]))
            self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        normal_init(self.content_encoder, std=0.001)
        if self.use_high_pass:
            normal_init(self.content_encoder2, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, x):
        """
        為適配 YOLO11，輸入為列表：[lr_feat, hr_feat]
        並返回 Concat 後的特徵以取代 Concat 層
        """
        lr_feat, hr_feat = x[0], x[1]
        
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)
        
        if self.semi_conv:
            if self.comp_feat_upsample:
                if self.use_high_pass:
                    mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
                    mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel, hamming=self.hamming_highpass)
                    compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat, mask_hr_init, self.highpass_kernel, self.up_group, 1)
                    
                    mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
                    mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel, hamming=self.hamming_lowpass)
                    
                    mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
                    mask_lr_lr_feat = F.interpolate(
                        carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

                    mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
                    mask_hr_lr_feat = F.interpolate(
                        carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel, self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
                    mask_hr = mask_hr_hr_feat + mask_hr_lr_feat
                else: 
                    raise NotImplementedError
            else:
                mask_lr = self.content_encoder(compressed_hr_feat) + F.interpolate(self.content_encoder(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
                if self.use_high_pass:
                    mask_hr = self.content_encoder2(compressed_hr_feat) + F.interpolate(self.content_encoder2(compressed_lr_feat), size=compressed_hr_feat.shape[-2:], mode='nearest')
        else:
            compressed_x = F.interpolate(compressed_lr_feat, size=compressed_hr_feat.shape[-2:], mode='nearest') + compressed_hr_feat
            mask_lr = self.content_encoder(compressed_x)
            if self.use_high_pass: 
                mask_hr = self.content_encoder2(compressed_x)
        
        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel, hamming=self.hamming_lowpass)
        if self.semi_conv:
                lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)
        else:
            lr_feat = F.interpolate(lr_feat, size=hr_feat.shape[2:], mode=self.upsample_mode)
            lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 1)

        if self.use_high_pass:
            mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel, hamming=self.hamming_highpass)
            hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
            if self.hr_residual:
                hr_feat = hr_feat_hf + hr_feat
            else:
                hr_feat = hr_feat_hf

        if self.feature_resample:
            lr_feat = self.dysampler(hr_x=compressed_hr_feat, lr_x=compressed_lr_feat, feat2sample=lr_feat)
                
        # YOLO 優化：直接在此完成拼接，取代原本的 Concat 層
        return torch.cat([lr_feat, hr_feat], dim=1)

# (LocalSimGuidedSampler 與 compute_similarity 維持原代碼即可，它們本身就是純 PyTorch)