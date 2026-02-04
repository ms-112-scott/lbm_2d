# Gemini Code Companion: lbm_2d

## 專案概述 (Project Overview)

`lbm_2d` 專案是一個基於 Taichi 語言的二維**計算流體力學（CFD）數據集生成工具**。它使用晶格波茲曼方法（Lattice Boltzmann Method, LBM）來模擬二維流場，並設計用於為深度學習研究，特別是 AI 代理模型（Surrogate Model）的訓練，提供大量的**動態（Time-series）流場數據**。

此專案的核心功能是批量處理，它可以自動讀取指定的遮罩（mask）圖像檔案作為幾何邊界，執行模擬，並將結果（如速度、密度、渦度）儲存為 `.npy` 格式的數據陣列，非常適合用於後續的 AI 模型訓練。

透過 Taichi 的大規模並行計算能力，求解器可以在 GPU 上高效運行，大幅縮短生成數據集所需的時間。

## 快速開始 (Quick Start)

本專案提供兩種模式來生成數據：

1.  **標準 2 通道數據 (速度):**
    執行 `main.py` 腳本，將生成包含 X 和 Y 方向速度 `(Vx, Vy)` 的數據。
    ```bash
    python main.py
    ```

2.  **多通道數據 (速度 + 密度 + 渦度):**
    執行 `main_multichannel.py` 腳本，將生成包含速度 `(Vx, Vy)`、密度 `(Rho)` 和渦度 `(Vorticity)` 的 **4 通道**數據，為 AI 模型提供更豐富的特徵。
    ```bash
    python main_multichannel.py
    ```

模擬的進度和結果將會顯示在終端機中，而輸出的檔案則會根據 `config.yaml` 的設定存放。

## 模擬組態 (`config.yaml`)

所有模擬的參數都在 `config.yaml` 檔案中進行設定。這是控制模擬行為的核心檔案。

**關鍵參數說明:**

-   `simulation`:
    -   `name`: 模擬任務的名稱，輸出的檔案會儲存在 `output/<name>` 目錄下。
    -   `niu`: 流體的黏滯係數，影響流場的雷諾數和穩定性。
    -   `save_npy`: `true` 表示儲存 `.npy` 陣列數據。
    -   `save_png`: `true` 表示儲存預覽圖像。
    -   `save_step`: 每隔多少步儲存一次數據。
-   `boundaries`:
    -   設定流場的邊界條件，如入口風速。
-   `obstacle`:
    -   `use_mask`: `true` 表示使用 PNG 圖像作為障礙物。
    -   `mask_dir`: 存放 `.png` 遮罩檔案的目錄。腳本會遍歷此目錄下的所有 PNG 檔案來進行批量模擬。

在執行模擬前，請務必檢查並修改 `config.yaml` 以符合您的研究需求。

## 專案結構 (Project Structure)

-   `main.py`: 標準 2 通道數據生成的入口腳本。
-   `engine.py`: 標準模式的核心模擬引擎。
-   `utils.py`: 標準模式的輔助工具函數。

-   `main_multichannel.py`: **(建議使用)** 4 通道數據生成的入口腳本。
-   `engine_multichannel.py`: 多通道模式的核心模擬引擎，包含計算密度和渦度的邏輯。
-   `utils_multichannel.py`: 多通道模式的輔助工具函數。

-   `config.yaml`: **(重要)** 全域配置文件。
-   `solver.py`: LBM 演算法的核心求解器，被 `engine` 所調用。
-   `cases/`: 存放幾何定義檔案，例如 `.stl` 或 `.txt`。
-   `output/masks/`: 預設存放用於生成模擬的 `.png` 遮罩圖。
-   `output/Building_Wind_Sim_multichannel/`: 預設的模擬結果輸出目錄。
-   `my_docs/`: 包含專案的說明文件和開發筆記。

## 輸出數據格式 (Output Data Format)

-   當 `save_npy` 設為 `true` 時，腳本會生成 `.npy` 檔案。
-   使用 `main_multichannel.py` 時，每個 `.npy` 檔案的數據形狀為 `(nx, ny, 4)`，其中 `nx` 和 `ny` 是網格尺寸，4 個通道分別代表：
    1.  **Vx**: X 方向速度
    2.  **Vy**: Y 方向速度
    3.  **Rho**: 流體密度
    4.  **Vorticity**: 渦度
