好# 數據集優化方案：使用 Zarr 高效管理 LBM 數據

本文件提出一個更高效的數據集管理方案，建議使用 [Zarr](https://zarr.readthedocs.io/en/stable/) 格式來取代傳統的單一 `.npy` 檔案儲存方式。此方案旨在解決大規模時序數據讀取時的 I/O 瓶頸，並提升數據處理的整體效率。

## 1. 為什麼選擇 Zarr？

相較於為每個時間步保存一個獨立的 `.npy` 檔案，Zarr 提供了多項關鍵優勢：

- **高效的 I/O 操作：** Zarr 將大型陣列分塊 (chunked) 儲存，並可對每個塊進行壓縮。這使得讀取數據的一個子集（例如幾個時間步）時，無需加載整個文件，從而極大地減少了讀取延遲和內存佔用。
- **並行處理：** Zarr 的設計原生支持並行讀寫，能夠充分利用現代多核 CPU 的性能，加速數據的預處理和加載過程。
- **內嵌 Metadata：** 模擬參數、數據描述等元數據可以直接作為屬性 (attributes) 存儲在 Zarr 的陣列或組中，使數據集完全自包含，無需依賴外部的 `metadata.csv`。
- **單一數據源：** 整個數據集（或一次模擬的所有數據）可以被視為單一的層級化存儲（一個目錄），簡化了數據管理和共享。

## 2. Zarr 數據集轉換流程

我們需要執行一個一次性的轉換腳本，將原始的 `.npy` 序列轉換為 Zarr 格式。

### 2.1 建議的 Zarr 結構

我們將创建一个根 `Group`，其中每個子 `Group` 代表一次完整的模擬實驗。

```
datasets/Taichi_LBM_v1.zarr/
├── .zgroup                 # Zarr root group file
├── run_cylinder_01/        # Sub-group for simulation 1
│   ├── .zgroup             # Sub-group file
│   ├── .zattrs             # Attributes for this simulation (e.g., simulation params)
│   ├── input_geometry      # Zarr array for input
│   │   ├── .zarray, .zattrs, 0.0.0, ...
│   └── output_sequence     # Zarr array for output time-series
│       ├── .zarray, .zattrs, 0.0.0.0.0, ...
└── run_building_complex_05/ # Sub-group for simulation 2
    ├── ...
```

### 2.2 數據轉換腳本範例

以下 Python 腳本展示了如何將 `.npy` 檔案轉換為上述 Zarr 結構。

```python
import os
import numpy as np
import zarr
from tqdm import tqdm

def convert_to_zarr(raw_data_dir: str, zarr_path: str):
    """
    將原始的 .npy 序列數據轉換為一個 Zarr 層級存儲。

    Args:
        raw_data_dir (str): 包含各次模擬運行的原始目錄 (e.g., 'output/Building_Wind_Sim')。
        zarr_path (str): 輸出的 Zarr 數據集路徑 (e.g., 'datasets/Taichi_LBM_v1.zarr')。
    """
    if os.path.exists(zarr_path):
        print(f"Zarr store at {zarr_path} already exists. Aborting.")
        return

    root_group = zarr.open_group(zarr_path, mode='w')

    simulation_runs = [d for d in os.listdir(raw_data_dir) if os.path.isdir(os.path.join(raw_data_dir, d))]

    for run_name in tqdm(simulation_runs, desc="Converting simulations to Zarr"):
        run_path = os.path.join(raw_data_dir, run_name)

        # 假設幾何檔案和 frames 目錄存在
        # 您需要根據您的實際結構調整這部分路徑
        input_geo_path = os.path.join(run_path, 'geometry.npy')
        frames_path = os.path.join(run_path, 'frames')

        if not os.path.exists(input_geo_path) or not os.path.exists(frames_path):
            print(f"Skipping {run_name}: missing geometry or frames directory.")
            continue

        # 1. 讀取輸入幾何
        input_geometry = np.load(input_geo_path)

        # 2. 讀取並堆疊時間序列
        frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.npy')])
        if not frame_files:
            continue

        sequence = []
        for frame_file in frame_files:
            frame_data = np.load(os.path.join(frames_path, frame_file))
            sequence.append(frame_data)
        output_sequence = np.stack(sequence, axis=0)

        # 3. 寫入 Zarr Group
        sim_group = root_group.create_group(run_name)
        sim_group.array('input_geometry', input_geometry, chunks=(64, 64, 64), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BIT))
        sim_group.array('output_sequence', output_sequence, chunks=(10, 1, 64, 64, 64), compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BIT))

        # 4. (可選) 添加元數據
        sim_group.attrs['original_path'] = run_path
        sim_group.attrs['num_frames'] = len(frame_files)

    print(f"Successfully created Zarr store at: {zarr_path}")

# --- 使用範例 ---
if __name__ == '__main__':
    # 假設您的原始數據已經整理成 `datasets/Taichi_LBM_v1_npy`
    # 這裡的 raw_data_dir 應指向上一版方案整理出的 npy 數據目錄
    raw_data_directory = 'datasets/Taichi_LBM_v1/simulations'
    zarr_output_path = 'datasets/Taichi_LBM_v1.zarr'

    # convert_to_zarr(raw_data_directory, zarr_output_path)

```

_注意：_ `chunks` 參數對性能至關重要，需要根據您的數據維度和訪問模式進行調整。

## 3. 使用 Zarr 的 PyTorch `Dataset`

讀取 Zarr 數據的 `Dataset` 類別變得更簡單、更乾淨。

```python
import torch
from torch.utils.data import Dataset, DataLoader
import zarr
from typing import Tuple

class ZarrLBMDataset(Dataset):
    """
    直接從 Zarr 存儲中讀取 LBM 動態流場數據。

    Args:
        zarr_path (str): Zarr 數據集的路徑 (e.g., 'datasets/Taichi_LBM_v1.zarr')。
    """
    def __init__(self, zarr_path: str):
        self.root_group = zarr.open_group(zarr_path, mode='r')
        self.simulation_keys = sorted(list(self.root_group.keys()))

    def __len__(self) -> int:
        """返回數據集中的樣本總數（即模擬次數）。"""
        return len(self.simulation_keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根據索引 `idx` 返回一個樣本的 Zarr 陣列。
        注意：這裡返回的是 Zarr Array 對象，實際的數據讀取是延遲的 (lazy)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - `input_geometry`: (1, D, H, W) 的張量。
            - `output_sequence`: (T, C, D, H, W) 的張量。
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        sim_key = self.simulation_keys[idx]
        sim_group = self.root_group[sim_key]

        # 獲取 Zarr 陣列
        input_zarr_array = sim_group['input_geometry']
        output_zarr_array = sim_group['output_sequence']

        # 從 Zarr 陣列讀取數據並轉換為 Tensor
        # `[:]` 觸發實際的數據讀取
        input_tensor = torch.from_numpy(input_zarr_array[:]).float().unsqueeze(0)
        output_tensor = torch.from_numpy(output_zarr_array[:]).float()

        return input_tensor, output_tensor

# --- 使用範例 ---
if __name__ == '__main__':
    zarr_dataset_path = 'datasets/Taichi_LBM_v1.zarr'

    # 確保 Zarr 數據集存在
    if not os.path.exists(zarr_dataset_path):
        print("Zarr dataset not found. Please run the conversion script first.")
    else:
        # 創建 Dataset 實例
        zarr_dataset = ZarrLBMDataset(zarr_path=zarr_dataset_path)

        # 驗證一個樣本
        if len(zarr_dataset) > 0:
            input_geo, output_seq = zarr_dataset[0]
            print(f"數據集大小: {len(zarr_dataset)}")
            print(f"輸入幾何張量形狀: {input_geo.shape}")
            print(f"輸出序列張量形狀: {output_seq.shape}")

        # 使用 DataLoader 進行批次處理
        dataloader = DataLoader(zarr_dataset, batch_size=4, shuffle=True, num_workers=2)

        for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
            print(f"\n--- Batch {batch_idx+1} ---")
            print(f"輸入批次形狀: {input_batch.shape}")
            print(f"輸出批次形狀: {output_batch.shape}")
            break
```

## 結論

採用 Zarr 不僅能解決當前數據集的 I/O 效率問題，還能為未來更大規模的數據生成和模型訓練提供一個可擴展、高性能的數據基礎設施。強烈建議您採納此方案。
