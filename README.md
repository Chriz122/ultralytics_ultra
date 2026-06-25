# Ultralytics Ultra - 模型大大大禮包

## 使用說明

支援版本
- 建議: `ultralytics==8.4.6`
-- 安裝補充相依套件：
  ```powershell
  # 安裝專案附帶的 wheel
  pip install Install_Dependencies\ultralytics-8.4.6.18.127-py3-none-any.whl
  # A2Mamba 依賴
  pip install Install_Dependencies\selective_scan_rh-0.0.0-cp311-cp311-linux_x86_64.whl
  https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.1/natten-0.21.1+torch280cu128-cp311-cp311-linux_x86_64.

  ```
  
  > [!IMPORTANT]
  > - `pytorch`、`torchvision`、`torchaudio` 請務必依你使用的 Python 版本與硬體（CUDA 或 CPU）選擇對應的套件版本。
  > - 請參考官方安裝說明來產生正確的安裝指令： https://pytorch.org/get-started/locally/
  > - 範例（僅示意；請以官方產出的指令為準）：
  >   - CPU-only：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
  >   - CUDA 11.8：`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  > - 版本建議 : Python 3.11、CUDA 12.8、PyTorch 2.8.0
  
此專案提供經過整理與擴充的 ultralytics 模型設定與範例，方便用於本機開發或替換 site-packages 中的 ultralytics 套件設定。

> [!IMPORTANT]
> 重要提醒：
> - 在替換系統套件前請務必先備份原始資料夾，並確認 Python 版本與相依套件相容。
> - 本 Repo 為 [ultralytics_pro](https://github.com/Chriz122/ultralytics_pro) 的 Linux 版本，建議於 Linux／WSL 環境執行，並支援 Mamba 系列等模型架構。

> [!TIP]
> ## 可搭配 YOLO_tools 的使用說明
> 可以搭配 [YOLO_tools](https://github.com/Chriz122/YOLO_tools) 的 toolbox 訓練、標註處理、評估等工作。