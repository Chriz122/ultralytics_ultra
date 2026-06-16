import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv

class BlockAttnResMerge(nn.Module):
    """
    增強版 Block Attention Residuals 融合模塊。
    支援融合任意數量的歷史層次，並自動處理不同層次間的空間解析度(H, W)差異。
    """
    def __init__(self, ch):
        """
        Args:
            ch (list[int]): 要融合的各個輸入特徵圖的通道數列表。
        """
        super().__init__()
        # 以傳入的第一個特徵分支（Anchor）作為基準通道數
        self.c_out = ch[0] 
        
        # 1x1 Conv 投影：統一所有分支的通道數為 c_out
        self.projs = nn.ModuleList([
            nn.Conv2d(c, self.c_out, 1, bias=False) if c != self.c_out else nn.Identity()
            for c in ch
        ])
        
        # 論文核心：可學習的單一 query 向量，必須初始化為 0
        self.query = nn.Parameter(torch.zeros(self.c_out))
        
        # RMSNorm 參數
        self.eps = 1e-6
        self.norm_weight = nn.Parameter(torch.ones(self.c_out))

    def forward(self, x: list[torch.Tensor]):
        """
        Args:
            x (list[torch.Tensor]): 來自前面多個網路層的特徵圖列表。
                                    第一個 tensor x[0] 被視為 Anchor，決定輸出的空間大小。
        """
        # 1. 空間對齊 (Spatial Alignment)
        target_size = x[0].shape[2:]  # 取得基準的 (H, W)
        aligned_x = []
        for xi in x:
            if xi.shape[2:] != target_size:
                # 若需要放大 (Upsample)，使用雙線性插值保留細節
                if xi.shape[2] < target_size[0]:
                    xi = F.interpolate(xi, size=target_size, mode='bilinear', align_corners=False)
                # 若需要縮小 (Downsample)，使用自適應平均池化提取大局特徵
                else:
                    xi = F.adaptive_avg_pool2d(xi, target_size)
            aligned_x.append(xi)
            
        # 2. 通道投影與堆疊 -> 形狀: [N分支, B批次, C通道, H高, W寬]
        V = torch.stack([proj(xi) for proj, xi in zip(self.projs, aligned_x)], dim=0)
        
        # 3. RMSNorm 計算
        variance = V.pow(2).mean(dim=2, keepdim=True)
        K = V * torch.rsqrt(variance + self.eps)
        K = K * self.norm_weight.view(1, 1, -1, 1, 1)
        
        # 4. 空間級 Attention Logits 計算
        # Query 尋找對應像素在不同層次特徵中的重要性
        logits = torch.einsum('c, n b c h w -> n b h w', self.query, K)
        
        # 5. 分支維度 Softmax 注意力分配
        attn_weights = logits.softmax(dim=0)
        
        # 6. 加權求和 (取代 Concat)
        out = torch.einsum('n b h w, n b c h w -> b c h w', attn_weights, V)
        
        return out


class Bottleneck_AttnRes(nn.Module):
    """
    改進版 Bottleneck: 
    使用論文的 Attention Residuals 取代標準的相加殘差 (x + f(x))。
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

        if self.add:
            # 論文參數：為殘差連接建立獨立的 Query 與 RMSNorm
            self.query = nn.Parameter(torch.zeros(c2))  # 初始化為0保證初期穩定
            self.norm_weight = nn.Parameter(torch.ones(c2))
            self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.add:
            # v0 = 原始輸入 (Identity), v1 = 卷積變換後的特徵
            v0 = x
            v1 = self.cv2(self.cv1(x))
            V = torch.stack([v0, v1], dim=0)  # Shape: [2, B, C, H, W]

            # RMSNorm 處理 (論文公式 K = RMSNorm(V))
            variance = V.pow(2).mean(dim=2, keepdim=True)
            K = V * torch.rsqrt(variance + self.eps)
            K = K * self.norm_weight.view(1, 1, -1, 1, 1)

            # 計算空間級 Attn Logits 並 Softmax
            logits = torch.einsum('c, n b c h w -> n b h w', self.query, K)
            attn_weights = logits.softmax(dim=0)

            # 注意力加權融合 (取代原本的 v0 + v1)
            out = torch.einsum('n b h w, n b c h w -> b c h w', attn_weights, V)
            return out
        else:
            return self.cv2(self.cv1(x))


class C2f_FullAttnRes(nn.Module):
    """
    改進版 C2f (Full Attention Residuals C2f):
    完美實現論文公式 (4)： h_l = \sum \alpha_{i \to l} * v_i
    捨棄原本的 torch.cat，讓每一個內部的 Bottleneck 都能動態回顧前面的所有歷史狀態。
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        
        # 【修改點 1】原本 cv2 接收 concat 膨脹後的通道 (2+n)*c，
        # 現在因為我們使用 AttnRes 融合所有歷史，通道數恆定為 self.c，參數大幅減少！
        self.cv2 = Conv(self.c, c2, 1)

        # 內部 Bottleneck 序列。因為 C2f 會使用 Full AttnRes 負責全局歷史融合，
        # 所以強制內部 Bottleneck 關閉自己的 shortcut，專心當純粹的特徵轉換器 f_l()
        self.m = nn.ModuleList(
            Bottleneck_AttnRes(self.c, self.c, shortcut=False, g=g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        )

        # 【核心設計】為每一層準備獨立的 Query
        # n 個 bottleneck 需要 n 個 query，最後的總融合還需要 1 個 query，共 n+1 組。
        self.queries = nn.ParameterList([nn.Parameter(torch.zeros(self.c)) for _ in range(n + 1)])
        self.norm_weights = nn.ParameterList([nn.Parameter(torch.ones(self.c)) for _ in range(n + 1)])
        self.eps = 1e-6

    def _attn_res_merge(self, v_list: list[torch.Tensor], step_idx: int) -> torch.Tensor:
        """根據論文封裝的 Full AttnRes 運算"""
        V = torch.stack(v_list, dim=0)  # [N歷史, B批次, C通道, H高, W寬]
        
        # 1. RMSNorm
        variance = V.pow(2).mean(dim=2, keepdim=True)
        K = V * torch.rsqrt(variance + self.eps)
        K = K * self.norm_weights[step_idx].view(1, 1, -1, 1, 1)

        # 2. Attention weight calculation
        logits = torch.einsum('c, n b c h w -> n b h w', self.queries[step_idx], K)
        attn = logits.softmax(dim=0)

        # 3. Weighted aggregation
        out = torch.einsum('n b h w, n b c h w -> b c h w', attn, V)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Full Attention Residuals."""
        # 1. 初始切割，產生 v0, v1 (類似論文的 Token Embedding 角色)
        y = list(self.cv1(x).chunk(2, 1))
        
        # V_history 保存所有歷史輸出 [v0, v1, v2, v3...]
        V_history = [y[0], y[1]]

        # 2. Full AttnRes 遞迴展開
        for i, m_layer in enumerate(self.m):
            # 論文: h_l = AttnRes(前 l-1 層歷史)
            h_l = self._attn_res_merge(V_history, step_idx=i)
            
            # 論文: v_l = f_l(h_l)
            v_l = m_layer(h_l)
            
            # 將新產生的特徵加入歷史中
            V_history.append(v_l)

        # 3. 最終匯總：取代原本粗暴的 torch.cat(y, 1)
        final_h = self._attn_res_merge(V_history, step_idx=self.n if hasattr(self, 'n') else len(self.m))

        return self.cv2(final_h)

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """與 forward 邏輯完全一致，只是替換為 split 運算"""
        y = self.cv1(x).split((self.c, self.c), 1)
        V_history = [y[0], y[1]]

        for i, m_layer in enumerate(self.m):
            h_l = self._attn_res_merge(V_history, step_idx=i)
            v_l = m_layer(h_l)
            V_history.append(v_l)

        final_h = self._attn_res_merge(V_history, step_idx=len(self.m))
        return self.cv2(final_h)