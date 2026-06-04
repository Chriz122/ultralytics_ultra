import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from torch.jit import Final


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, patch_stride=4, patch_pad=0, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=patch_pad)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # 輸入為 [B, H, W, C]，轉換為 [B, C, H, W] 給 Conv2d 處理，再轉回 [B, H', W', C']
        x = self.proj(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class AttMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    fast_attn: Final[bool]

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(dim // num_heads, 32)
        self.scale = self.head_dim ** -0.5
        self.fast_attn = hasattr(torch.nn.functional, 'scaled_dot_product_attention')  

        self.qkv = nn.Linear(dim, self.num_heads * self.head_dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AttentionBlock(nn.Module):
    def __init__(
            self,
            dim, mlp_ratio=4., num_heads=8, qkv_bias=False, qk_norm=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = AttMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.size()
        x = x.reshape(B, H * W, C).contiguous()
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, C).contiguous()
        return x


class ContextLayer(nn.Module):
    def __init__(self, in_dim, conv_dim, context_size=[3], context_act=nn.GELU,
                 context_f=True, context_g=True):
        # channel last
        super().__init__()
        self.f = nn.Linear(in_dim, conv_dim) if context_f else nn.Identity()
        self.g = nn.Linear(conv_dim, in_dim) if context_g else nn.Identity()
        self.context_size = context_size
        self.act = context_act() if context_act else nn.Identity()
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size]
        self.context_list = nn.ModuleList()
        for c_size in context_size:
            self.context_list.append(
                nn.Conv2d(conv_dim, conv_dim, c_size, stride=1, padding=c_size // 2, groups=conv_dim)
            )

    def forward(self, x):
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        out = 0
        for i in range(len(self.context_list)):
            ctx = self.act(self.context_list[i](x))
            out = out + ctx
        out = self.g(out.permute(0, 2, 3, 1).contiguous())
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0., bias=True, conv_in_mlp=True,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.conv_in_mlp = conv_in_mlp
        if self.conv_in_mlp:
            self.conv_group_dim = conv_group_dim
            self.conv_dim = hidden_features // conv_group_dim
            self.context_layer = ContextLayer(in_features, self.conv_dim, context_size,
                                              context_act, context_f, context_g)

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        if hidden_features == in_features and conv_group_dim == 1:
            self.expand_dim = False
        else:
            self.expand_dim = True
            self.act = act_layer()
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.conv_in_mlp:
            conv_x = self.context_layer(x)
        x = self.fc1(x)
        if self.expand_dim:
            x = self.act(x)
            x = self.drop(x)
        if self.conv_in_mlp:
            if self.expand_dim:
                x = x * conv_x.repeat(1, 1, 1, self.conv_group_dim)
            else:
                x = x * conv_x
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True
                 ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop, bias=bias, conv_in_mlp=conv_in_mlp,
                       conv_group_dim=conv_group_dim, context_size=context_size, context_act=context_act,
                       context_f=context_f, context_g=context_g)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.gamma_1 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = shortcut + self.drop_path(self.gamma_1 * self.mlp(self.norm(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, depth,
                 mlp_ratio=4., att_ratio=4., conv_in_mlp=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 bias=True, use_layerscale=False, layerscale_value=1e-4,
                 conv_group_dim=4, context_size=3, context_act=nn.GELU,
                 context_f=True, context_g=True,
                 downsample=None, patch_size=3, patch_stride=2, patch_pad=1, patch_norm=True,
                 attention_depth=0):

        super().__init__()
        self.dim = dim
        self.depth = depth
        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * depth
        if not isinstance(conv_group_dim, (list, tuple)):
            conv_group_dim = [conv_group_dim] * depth
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size] * depth
            
        self.blocks = nn.ModuleList([
            BasicBlock(
                dim=dim, mlp_ratio=mlp_ratio[i], conv_in_mlp=conv_in_mlp, drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                act_layer=act_layer, norm_layer=norm_layer,
                bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                conv_group_dim=conv_group_dim[i], context_size=context_size[i], context_act=context_act,
                context_f=context_f, context_g=context_g
            )
            for i in range(depth)])

        if attention_depth > 0:
            for j in range(attention_depth):
                self.blocks.append(AttentionBlock(
                    dim=dim, mlp_ratio=att_ratio, drop=drop, drop_path=drop_path[depth + j],
                    act_layer=act_layer, norm_layer=norm_layer,
                ))

        if downsample is not None:
            self.downsample = downsample(
                in_chans=dim,
                embed_dim=out_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_pad=patch_pad,
                norm_layer=norm_layer if patch_norm else None
            )
        else:
            self.downsample = None

    def forward(self, x):
        # 通過所有的 Block 計算
        for blk in self.blocks:
            x = blk(x)
        
        # 保存未進行 downsample 時的特徵，供 YOLO 提取
        out = x 
        
        if self.downsample is not None:
            x = self.downsample(x)
        # 回傳: [提供給外部的當前層特徵], [準備輸入給下一層的特徵]
        return out, x


class EfficientMod(nn.Module):
    def __init__(self,
                 img_size=224, # 為了與 SMT 統一新增
                 in_chans=3, num_classes=1000,
                 patch_size=[4, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[0, 1, 1, 1], patch_norm=True,
                 embed_dim=[64, 128, 256, 512], depths=[2, 2, 6, 2], attention_depth=[0, 0, 0, 0],
                 mlp_ratio=[4.0, 4.0, 4.0, 4.0], att_ratio=[4, 4, 4, 4],
                 conv_in_mlp=[True, True, True, True],
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=False, layerscale_value=1e-4,
                 bias=True, drop_rate=0., drop_path_rate=0.0,
                 conv_group_dim=[4, 4, 4, 4], context_size=[3, 3, 3, 3], context_act=nn.GELU,
                 context_f=True, context_g=True,
                 **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.img_size = img_size
        self.in_chans = in_chans
        self.depths = depths
        self.attention_depth = attention_depth
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]

        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            patch_size=patch_size[0],
            patch_stride=patch_stride[0],
            patch_pad=patch_pad[0],
            norm_layer=norm_layer if self.patch_norm else None)

        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, (sum(depths) + sum(attention_depth)))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim[i_layer],
                               out_dim=embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               depth=depths[i_layer],
                               mlp_ratio=mlp_ratio[i_layer],
                               att_ratio=att_ratio[i_layer],
                               conv_in_mlp=conv_in_mlp[i_layer],
                               drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]) + sum(attention_depth[:i_layer]):sum(
                                   depths[:i_layer + 1]) + sum(attention_depth[:i_layer + 1])],
                               act_layer=act_layer, norm_layer=norm_layer,
                               bias=bias, use_layerscale=use_layerscale, layerscale_value=layerscale_value,
                               conv_group_dim=conv_group_dim[i_layer],
                               context_size=context_size[i_layer],
                               context_act=context_act,
                               context_f=context_f,
                               context_g=context_g,
                               downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                               patch_size=patch_size[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               patch_stride=patch_stride[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               patch_pad=patch_pad[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                               patch_norm=patch_norm,
                               attention_depth=attention_depth[i_layer]
                               )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        
        # 分類頭暫時保留供分類任務使用，但在 YOLO(Backbone) 中不使用它
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # ---------------- 增加 self.width_list 的計算 (同 SMT) ----------------
        self.width_list = []
        try:
            self.eval() 
            dummy_input = torch.randn(1, self.in_chans, self.img_size, self.img_size)
            with torch.no_grad():
                features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
            self.train() 
        except Exception as e:
            print(f"Error during dummy forward pass for width_list calculation: {e}")
            self.width_list = self.embed_dim 
            self.train()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        # 最初輸入需從 [B, C, H, W] 轉換為 [B, H, W, C] 
        x = self.patch_embed(x.permute(0, 2, 3, 1).contiguous())
        
        feature_outputs = [] # List of Tensors 解決 "Insert error"
        
        for i, layer in enumerate(self.layers):
            out, x = layer(x)
            
            # 對於最後一層，施加 self.norm
            if i == len(self.layers) - 1:
                out = self.norm(out)
                
            # 將格式轉回 YOLO 與 Pytorch 預設的空間排佈格式 [B, C, H, W]
            out_spatial = out.permute(0, 3, 1, 2).contiguous()
            feature_outputs.append(out_spatial)
            
        return feature_outputs 

    def forward(self, x):
        # 回傳 List 提供給目標檢測模型(YOLO)對接
        features = self.forward_features(x)
        return features


# ===================== 工廠函數 (與 SMT 格式對齊) =====================

@register_model
def efficientMod_xxs(pretrained=False, img_size=224, **kwargs):
    depths = [2, 2, 6, 2]
    attention_depth = [0, 0, 1, 2]
    att_ratio = [0, 0, 4, 4]
    mlp_ratio = [[1, 6, 1, 6], [1, 6, 1, 6], [1, 6] * 3, [1, 6, 1, 6]]
    context_size = [[7] * 10, [7] * 10, [7] * 20, [7] * 10]
    conv_group_dim = mlp_ratio
    
    model = EfficientMod(img_size=img_size, in_chans=3, num_classes=1000,
                         patch_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[3, 1, 1, 1], patch_norm=True,
                         embed_dim=[32, 64, 128, 256], depths=depths, attention_depth=attention_depth,
                         mlp_ratio=mlp_ratio, att_ratio=att_ratio,
                         conv_in_mlp=[True, True, True, True],
                         act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True, layerscale_value=1e-4,
                         bias=True, drop_rate=0., drop_path_rate=0.0,
                         conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                         context_f=True, context_g=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def efficientMod_xs(pretrained=False, img_size=224, **kwargs):
    depths = [3, 3, 4, 2]
    attention_depth = [0, 0, 3, 3]
    att_ratio = [4, 4, 4, 4]
    mlp_ratio = [[1, 4, 1, 4] * 4, [1, 4, 1, 4] * 4, [1, 4, 1, 4] * 10, [1, 4, 1, 4] * 4]
    context_size = [[7] * 10, [7] * 10, [7] * 20, [7] * 10]
    conv_group_dim = mlp_ratio
    
    model = EfficientMod(img_size=img_size, in_chans=3, num_classes=1000,
                         patch_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[3, 1, 1, 1], patch_norm=True,
                         embed_dim=[32, 64, 144, 288], depths=depths, attention_depth=attention_depth,
                         mlp_ratio=mlp_ratio, att_ratio=att_ratio,
                         conv_in_mlp=[True, True, True, True],
                         act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True, layerscale_value=1e-4,
                         bias=True, drop_rate=0., drop_path_rate=0.00,
                         conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                         context_f=True, context_g=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def efficientMod_s(pretrained=False, img_size=224, **kwargs):
    depths = [4, 4, 8, 4]
    attention_depth = [0, 0, 4, 4]
    att_ratio = [4, 4, 4, 5]
    mlp_ratio = [[1, 6, 1, 6] * 4, [1, 6, 1, 6] * 4, [1, 6, 1, 6] * 10, [1, 6, 1, 6] * 4]
    context_size = [[7] * 10, [7] * 10, [7] * 20, [7] * 10]
    conv_group_dim = mlp_ratio
    
    model = EfficientMod(img_size=img_size, in_chans=3, num_classes=1000,
                         patch_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[3, 1, 1, 1], patch_norm=True,
                         embed_dim=[32, 64, 144, 312], depths=depths, attention_depth=attention_depth,
                         mlp_ratio=mlp_ratio, att_ratio=att_ratio,
                         conv_in_mlp=[True, True, True, True],
                         act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True, layerscale_value=1e-4,
                         bias=True, drop_rate=0., drop_path_rate=0.02,
                         conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                         context_f=True, context_g=True, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def efficientMod_s_Conv(pretrained=False, img_size=224, **kwargs):
    depths = [4, 4, 12, 8]
    mlp_ratio = [[1, 6, 1, 6, 1, 6], [1, 6, 1, 6, 1, 6], [1, 6, 1, 6] * 5, [1, 6] * 8]
    context_size = [[7] * 10, [7] * 10, [7] * 20, [7] * 12]
    conv_group_dim = mlp_ratio
    
    model = EfficientMod(img_size=img_size, in_chans=3, num_classes=1000,
                         patch_size=[7, 3, 3, 3], patch_stride=[4, 2, 2, 2], patch_pad=[3, 1, 1, 1], patch_norm=True,
                         embed_dim=[40, 80, 160, 344], depths=depths, mlp_ratio=mlp_ratio,
                         conv_in_mlp=[True, True, True, True],
                         act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layerscale=True, layerscale_value=1e-4,
                         bias=True, drop_rate=0., drop_path_rate=0.02,
                         conv_group_dim=conv_group_dim, context_size=context_size, context_act=nn.GELU,
                         context_f=True, context_g=True, **kwargs)
    model.default_cfg = _cfg()
    return model

# ========= 測試確保符合 YOLO List 與 Width要求 =========
if __name__ == '__main__':
    img_h, img_w = 640, 640  # 模擬 YOLO 輸入尺寸
    print("--- 建立 EfficientMod XXS 模型 ---")
    model = efficientMod_xxs(img_size=img_h)
    print("模型建立成功。")
    print("內部自動計算的 width_list:", model.width_list)

    input_tensor = torch.rand(2, 3, img_h, img_w)
    print(f"\n--- 測試正向傳播 (輸入尺寸: {input_tensor.shape}) ---")

    model.eval()
    with torch.no_grad():
        output_features = model(input_tensor)
        
    print("正向傳播成功，目前模型輸出改為 YOLO 需要的 List 格式:")
    for i, features in enumerate(output_features):
        print(f"Stage {i+1} 輸出維度: {features.shape}") 

    runtime_widths = [f.size(1) for f in output_features]
    print("\n實際運行的通道數:", runtime_widths)
    assert model.width_list == runtime_widths, "通道數列表不匹配!"
    print("width_list 驗證通過，現在這個架構不會在 YOLO 報 Tensor object has no attribute insert 的錯了。")