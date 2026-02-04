已為您更新代辦事項，並針對 **Zarr 存儲格式** 的實作細節重新整理了 TODO 文件。這將確保您的 `taichi_LBM3D` 專案在數據生成階段就能具備高性能的 I/O 能力，為後續的 m-NCA 訓練打下基礎。

---

# 📝 TODO：實現基於 Zarr 的 LBM 數據集批量生成

## 階段一：核心求解器與 Zarr 整合 (`lbm_solver_3d.py`)

**目標：** 替換傳統 `.npy`，使用 Zarr 實現高效、分塊（Chunked）的 5D 張量儲存。

- [ ] **1. 更新 `config.yml` 儲存參數：**
- 新增 `zarr_output: true` 開關。
- 定義 `chunks` 大小（建議如 `[1, 64, 64, 64, 4]`，有利於 NCA 局部讀取）。
- 設定 `compression` 等級（如使用 `lz4` 或 `zstd`）。

- [ ] **2. 實作 `ZarrWriter` 類別：**
- 在求解器中建立一個初始化函數，根據模擬總步長預先分配（Pre-allocate）Zarr 陣列空間。
- **[關鍵]** 確保維度順序為 `(T, D, H, W, C)` 以符合 PyTorch 慣例。

- [ ] **3. 整合至主迴圈：**
- 在 `iter % numpy_output_frequency == 0` 時，調用 `zarr_array[frame_idx] = field.to_numpy()`。
- **優化建議：** 考慮使用 `ti.sync()` 確保數據一致性後再進行寫入。

## 階段二：建立批量執行與數據匯總腳本 (`batch_runner.py`)

**目標：** 自動遍歷多個幾何案例，並生成對應的 Zarr 數據集結構。

- [ ] **1. 自動化路徑管理：**
- 為每個案例創建獨立的 `.zarr` 資料夾（Group）。
- 在根目錄生成一個 `dataset_summary.json`，記錄每個案例的幾何參數、Reynolds 數與對應的 Zarr 路徑。

- [ ] **2. 幾何文件自動載入：**
- 修改腳本使其能從 `geometry/` 資料夾中讀取多個圖片或 `.dat` 檔。
- 動態修改 `config_run.yml` 並調用 `subprocess` 運行模擬。

- [ ] **3. 異常處理機制：**
- 若模擬中途崩潰（如數值發散），需確保已寫入的 Zarr 數據不會損壞，並記錄錯誤日誌跳至下一個案例。

## 階段三：ML 讀取接口驗證 (DataLoader)

**目標：** 確保生成的 Zarr 格式能被 PyTorch 高效讀取。

- [ ] **1. 撰寫 `LBMDataset(Dataset)` 類別：**
- 使用 `zarr.open(mode='r')` 實現延遲加載（Lazy Loading）。

- [ ] **2. 效能測試：**
- 對比 `npy` 讀取與 `zarr` 讀取在訓練時的 CPU/GPU 佔用率。

---

### 💡 實作備忘錄 (Zarr Tips)：

- **數據類型：** 除非精確度要求極高，否則建議儲存為 `float32` 以節省一半空間。
- **併發寫入：** 由於 Taichi 是單進程運行，直接寫入是安全的；若未來改為多進程生成，需注意 Zarr 的 `ProcessSafe` 設定。
