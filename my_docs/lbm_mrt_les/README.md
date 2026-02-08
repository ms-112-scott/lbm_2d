# LBM MRT-LES 專案文檔

本文件記錄 `src/lbm_mrt_les` 目錄下的專案結構與程式碼邏輯分析。此專案實作了一個基於 Taichi 的二維晶格波茲曼方法 (LBM) 流體模擬器，採用多重鬆弛時間 (MRT) 碰撞模型與大渦模擬 (LES) 湍流模型。

## 1. 專案概觀 (Project Overview)

此模擬器旨在解決高雷諾數下的二維流場問題，特別針對室內氣流 (Room Jet Flow) 或圓柱繞流進行模擬。

- **核心算法**: Lattice Boltzmann Method (LBM) D2Q9 模型。
- **碰撞模型**: Multi-Relaxation Time (MRT)，提供比單一鬆弛時間 (SRT/BGK) 更好的數值穩定性。
- **湍流模型**: Large Eddy Simulation (LES) with Smagorinsky model，用於處理高雷諾數下的次網格應力。
- **加速技術**: 使用 Taichi 語言進行 GPU 加速計算。
- **邊界處理**:
  - Sponge Layer (阻尼層): 在出口處增加黏滯性以吸收反射波。
  - Masking: 支援複雜幾何邊界 (如雙房間隔間、圓柱)。

## 2. 檔案結構與功能 (File Structure & Functions)

| 檔案名稱               | 功能說明                                                                                                                                                                               |
| :--------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`config.yaml`**      | **配置檔**。定義模擬參數 (網格大小、黏滯係數、時間步長)、物理模型參數 (LES 常數、MRT 鬆弛率)、邊界條件與障礙物遮罩設定。                                                               |
| **`lbm2d_mrt_les.py`** | **核心求解器 (Solver)**。包含 `LBM2D_MRT_LES` 類別。負責記憶體配置、MRT 矩陣定義、碰撞與串流 (Collide & Stream)、邊界條件應用以及 LES 湍流模型計算。                                   |
| **`main.py`**          | **主程式入口**。負責讀取配置、初始化 Solver、設定視覺化視窗 (GUI)、錄製影片以及執行主要的模擬迴圈。包含策略性的模擬步數計算 (基於 Flow-through time)。                                 |
| **`utils.py`**         | **工具函式庫**。包含配置讀取、雷諾數計算、以及幾何遮罩 (Mask) 的生成邏輯 (支援圓柱與可旋轉的雙房間佈局)。                                                                              |
| **`visualization.py`** | **視覺化模組**。包含 `LBMVisualizer` 類別。負責將 LBM 的速度場與渦度場轉換為 RGB 圖像，並處理顯示尺寸的縮放。                                                                          |
| **`VideoRecorder.py`** | **影片錄製模組**。封裝 `imageio_ffmpeg`，將模擬過程即時錄製為 MP4 影片。                                                                                                               |
| **`Rans.py`**          | **後處理工具 (Post-processing)**。雖然檔名為 RANS，但實際功能是讀取模擬輸出的影片檔，計算**時間平均 (Temporal Average)** 的流場影像，用於模擬雷諾平均 (Reynolds-Averaged) 的結果分析。 |

## 3. 核心邏輯詳解 (Core Logic Analysis)

### 3.1 MRT-LES 求解器 (`lbm2d_mrt_les.py`)

- **MRT 矩陣**: 使用標準 D2Q9 轉換矩陣 $M$，將分佈函數 $f$ 轉換到矩矩空間 $m$ ($m = M f$) 進行碰撞操作，再轉回 $f$ ($f = M^{-1} m$)。
- **LES Smagorinsky**: 在碰撞步驟中，根據非平衡態動量矩 (Non-equilibrium moments) 計算局部應變率，進而動態調整鬆弛時間 $\tau_{eff}$ (Effective Tau)，以模擬湍流黏滯性。
  - $\tau_{eddy} \propto \sqrt{Q}$ (Q 為動量通量)
- **Sponge Layer**: 在模擬區域右側 (出口附近) 設置阻尼層，透過二次曲線平滑增加黏滯係數，防止流體在出口處產生非物理的反射波。

### 3.2 配置系統 (`config.yaml`)

- **Simulation**: 控制 `nx`, `ny` (解析度), `niu` (黏滯性), `smagorinsky_constant` (LES 強度)。
- **Boundaries**: 定義四個邊界的類型 (Inlet/Outlet/Wall) 與數值。
- **Mask**: 支援 `room` (雙房間) 與 `cylinder` (圓柱) 兩種模式。`room` 模式在 `utils.py` 中實作了座標旋轉功能，可模擬不同風向角。

### 3.3 執行流程 (`main.py`)

1.  **Init**: 讀取 Config，建立 Mask，初始化 Solver。
2.  **Strategy**: 根據入口風速與網格寬度，自動計算 "Flow-through Time" (流體穿過一次所需時間)，並設定目標 Pass 數 (預設 5 Passes) 以決定總模擬步數。
3.  **Loop**:
    - 執行 LBM 運算 (GPU)。
    - 更新進度條 (tqdm)。
    - 渲染畫面 (Velocity/Vorticity) 並顯示於 GUI。
    - 寫入影片幀。
4.  **Finalize**: 結束錄影。

## 4. 使用說明 (Usage)

### 執行模擬

在專案根目錄下執行：

```bash
python src/lbm_mrt_les/main.py [config_path]
```

- 若未指定 `config_path`，預設讀取 `src/lbm_mrt_les/config.yaml`。

### 執行 RANS (時間平均) 分析

模擬完成後，可針對生成的影片進行時間平均分析：

```bash
python src/lbm_mrt_les/Rans.py
```

_(需自行修改腳本內的 `input_video` 路徑)_

## 5. 開發筆記與觀察

- **Rans.py 命名**: 該檔案目前的實作僅為「影片時間平均計算」，而非 RANS 湍流模型的求解器。其目的是從 LES 的瞬時結果中提取平均流場，以供 RANS 分析對比。
- **性能優化**: 專案已使用 Taichi Kernel 進行 GPU 加速，且 MRT 矩陣運算已預先計算反矩陣 ($M^{-1}$)，避免每步重複求逆。
- **Sponge Layer**: 目前硬編碼在 `lbm2d_mrt_les.py` 的 `collide_and_stream` kernel 中，寬度為 200 grid units。若需調整需修改原始碼。
