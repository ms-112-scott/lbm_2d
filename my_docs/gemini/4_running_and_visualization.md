# 5. 執行與視覺化

完成幾何準備與參數設定後，您就可以執行模擬並將結果視覺化。

## 執行模擬

我們已經將整個流程打包成一個簡單的 pipeline。

1.  **編輯設定檔:** 確認 `Single_phase/config.yml` 中的所有參數都已根據您的需求設定完成。特別是 `run_mesh` 和 `run_simulation` 旗標。

2.  **執行 pipeline:** 在您的終端機中，切換到 `Single_phase` 目錄下，然後執行：
    ```bash
    python run.py
    ```

腳本將會：
*   如果 `run_mesh` 為 `true`，則執行體素化。
*   如果 `run_simulation` 為 `true`，則執行 LBM 模擬。

所有的輸出檔案，包括體素化後的幾何檔案 (`geometry.txt`) 和模擬結果 (`.vtk` 檔案)，都將會儲存在 `cases/<您在 config.yml 中設定的案例名稱>/` 目錄下。

## 視覺化

模擬會產生一系列的 `.vtk` 檔案 (例如 `structured0.vtk`, `structured10000.vtk`, ...)。這些是 [VTK (Visualization Toolkit)](https://vtk.org/) 格式的檔案，您可以使用免費且強大的開源軟體 [ParaView](https://www.paraview.org/) 來開啟並分析它們。

**使用 ParaView 的基本步驟:**

1.  **開啟檔案:** 在 ParaView 中，選擇 `File -> Open...`，然後導航到 `cases/<您的案例名稱>/` 目錄，選擇您想要查看的 `.vtk` 檔案系列。ParaView 會自動將它們辨識為一個時間序列。
2.  **套用視覺化濾鏡:**
    *   **Contour:** 建立等值線或等值面，例如顯示特定風速的區域。
    *   **Glyph:** 在網格點上顯示向量（例如風速），可以用箭頭來表示風的方向與大小。
    *   **Stream Tracer:** 顯示流線，以觀察空氣的流動路徑。
    *   **Slice:** 建立切片，以查看模型內部的流場分佈。
3.  **調整顏色與標尺:** 根據您想要呈現的物理量 (如 `rho` 壓力, `velocity` 速度) 來調整顏色映射表與圖例。
4.  **播放動畫:** 使用 ParaView 的動畫控制項來播放整個時間序列，以觀察風場隨時間的變化。

透過 ParaView，您可以深入分析模擬結果，例如找出建築物周圍的風速加速區、靜風區、渦流區等。
