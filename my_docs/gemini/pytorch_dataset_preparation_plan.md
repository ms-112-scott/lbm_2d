# 將 LBM 模擬數據整理為 PyTorch 數據集

本文件旨在說明如何將 `taichi_LBM3D` 專案產生的原始模擬數據，整理成一個結構清晰、易於使用的 PyTorch 數據集，以利後續的 AI 代理模型訓練。

## 1. 數據結構分析與假設

根據專案目標，我們假設原始數據儲存在 `output/Building_Wind_Sim/` 目錄下，且具有以下結構：每個子目錄代表一次獨立的模擬，其中包含定義邊界的幾何檔案和一系列代表動態流場的時間序列檔案。

**原始結構範例 (假設):**

```
output/Building_Wind_Sim/
├── run_cylinder_01/
│   ├── geometry.npy
│   ├── frame_0000.npy
│   ├── frame_0001.npy
│   └── ...
└── run_building_complex_05/
    ├── geometry.npy
    ├── frame_0000.npy
    ├── frame_0001.npy
    └── ...
```

## 2. 數據集整理策略

我們將建立一個新的、機器學習友好的目錄結構，並通過一個 `metadata.csv` 檔案來索引所有數據。

**目標結構:**

```
datasets/Taichi_LBM_v1/
├── metadata.csv
└── simulations/
    ├── run_cylinder_01/
    │   ├── geometry.npy
    │   └── frames/
    │       ├── 0000.npy
    │       ├── 0001.npy
    │       └── ...
    └── run_building_complex_05/
        ├── geometry.npy
        └── frames/
            ├── 0000.npy
            ├── 0001.npy
            └── ...
```

### 2.1. Metadata 檔案 (`metadata.csv`)

這個 CSV 檔案是數據集的核心，它將幾何輸入與時間序列輸出關聯起來。

**`metadata.csv` 格式範例:**

```csv
input_path,output_path
simulations/run_cylinder_01/geometry.npy,simulations/run_cylinder_01/frames
simulations/run_building_complex_05/geometry.npy,simulations/run_building_complex_05/frames
```

### 2.2. 整理腳本 (建議)

建議撰寫一個 Python 腳本來自動化這個過程。腳本應能：
1. 遍歷 `output/Building_Wind_Sim/` 下的所有模擬。
2. 在 `datasets/Taichi_LBM_v1/simulations/` 中為每次模擬創建對應的新目錄結構。
3. 複製並重命名檔案。
4. 生成 `metadata.csv` 檔案。

## 3. PyTorch `Dataset` 實作

以下是一個自定義的 PyTorch `Dataset` 類別範本，它可以讀取我們剛剛定義的數據集結構。

**`LBMDataset.py` 程式碼範例:**

```python
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class LBMDataset(Dataset):
    """
    用於 Taichi LBM 動態流場數據集的 PyTorch Dataset 類別。

    Args:
        root_dir (str): 數據集根目錄 (例如 'datasets/Taichi_LBM_v1')。
        metadata_file (str): metadata CSV 檔案名稱 (預設為 'metadata.csv')。
    """
    def __init__(self, root_dir: str, metadata_file: str = 'metadata.csv'):
        self.root_dir = root_dir
        metadata_path = os.path.join(self.root_dir, metadata_file)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
        
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self) -> int:
        """返回數據集中的樣本總數。"""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根據索引 `idx` 加載一個樣本（輸入幾何和輸出時間序列）。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - `input_geometry`: (1, D, H, W) 的張量，代表輸入幾何。
            - `output_sequence`: (T, C, D, H, W) 的張量，代表流場時間序列。
                                T 是時間步長，C 是物理量通道數（如速度u, v, w）。
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
            
        # 獲取檔案路徑
        input_rel_path = self.metadata.iloc[idx]['input_path']
        output_rel_path = self.metadata.iloc[idx]['output_path']
        
        input_abs_path = os.path.join(self.root_dir, input_rel_path)
        output_abs_path = os.path.join(self.root_dir, output_rel_path)

        # 1. 加載輸入幾何
        input_geometry = np.load(input_abs_path)
        # 增加通道維度 (C, D, H, W) -> (1, D, H, W)
        input_tensor = torch.from_numpy(input_geometry).float().unsqueeze(0)

        # 2. 加載輸出時間序列
        frame_files = sorted([f for f in os.listdir(output_abs_path) if f.endswith('.npy')])
        
        sequence = []
        for frame_file in frame_files:
            frame_path = os.path.join(output_abs_path, frame_file)
            frame_data = np.load(frame_path) # 假設形狀為 (C, D, H, W)
            sequence.append(frame_data)
        
        # 將 NumPy 陣列堆疊成一個大的 NumPy 陣列 (T, C, D, H, W)
        output_sequence = np.stack(sequence, axis=0)
        output_tensor = torch.from_numpy(output_sequence).float()
        
        return input_tensor, output_tensor

# --- 使用範例 ---
if __name__ == '__main__':
    # 假設您已經根據上述結構整理好了數據集
    dataset_root = 'datasets/Taichi_LBM_v1' # 您的數據集路徑
    
    # 創建 Dataset 實例
    lbm_dataset = LBMDataset(root_dir=dataset_root)
    
    # 驗證一個樣本
    if len(lbm_dataset) > 0:
        input_geo, output_seq = lbm_dataset[0]
        print(f"數據集大小: {len(lbm_dataset)}")
        print(f"輸入幾何張量形狀: {input_geo.shape}")   # 應為 [1, D, H, W]
        print(f"輸出序列張量形狀: {output_seq.shape}") # 應為 [T, C, D, H, W]

    # 使用 DataLoader 進行批次處理
    dataloader = DataLoader(lbm_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 迭代一個 epoch
    for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
        print(f"\n--- Batch {batch_idx+1} ---")
        print(f"輸入批次形狀: {input_batch.shape}")
        print(f"輸出批次形狀: {output_batch.shape}")
        # 在這裡將數據送入您的模型進行訓練
        # model(input_batch, output_batch)
        break # 只演示一個批次
```

## 4. 下一步

1.  **實施整理腳本：** 根據您的具體檔名規則，撰寫 Python 腳本將 `output/Building_Wind_Sim` 的內容轉換為 `datasets/Taichi_LBM_v1` 的結構。
2.  **整合 `LBMDataset`：** 將 `LBMDataset.py` 放入您的機器學習專案中。
3.  **開始訓練：** 使用 `DataLoader` 加載數據，並開始訓練您的 PyTorch 模型。

這個流程能確保您的數據集管理起來既高效又有條理，為後續的研究奠定穩固的基礎。
