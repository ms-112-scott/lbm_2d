# 2. 案例管理與配置

為了更好地組織您的模擬結果並提高可重複性，我們導入了案例管理系統，並將所有可配置的參數集中到一個 YAML 格式的檔案 `Single_phase/config.yml` 中。

## 配置檔案 (`config.yml`) 結構

`config.yml` 檔案位於 `Single_phase/` 目錄下，包含以下主要區塊：

```yaml
# 案例設定
case_name: my_first_case              # 案例名稱，將作為輸出資料夾的名稱 (例如：cases/my_first_case)

# 階段式執行控制
run_mesh: true                      # 是否執行網格生成階段 (voxelize_geometry.py)
run_simulation: true                # 是否執行模擬階段 (lbm_solver_3d.py)

# 幾何相關參數
geometry:
  stl_file: path_to_your_stl_file.stl  # 輸入的 STL 檔案路徑 (相對於 Single_phase/ 目錄)
  output_file: geometry.txt           # 輸出的體素化幾何檔案名稱 (將儲存在案例資料夾中)
  voxel_size: 0.5                     # 體素大小 (解析度)

# 模擬相關參數
simulation:
  input_file: geometry.txt            # 輸入的體素化幾何檔案名稱 (將從案例資料夾中讀取)
  nx: 100                             # 網格 X 方向大小 (會由 voxelize_geometry.py 自動更新)
  ny: 80                              # 網格 Y 方向大小 (會由 voxelize_geometry.py 自動更新)
  nz: 50                              # 網格 Z 方向大小 (會由 voxelize_geometry.py 自動更新)
  niu: 0.01                           # 流體黏滯係數
  fx: 1.0e-5                          # X 方向的外部力
  fy: 0.0                             # Y 方向的外部力
  fz: 0.0                             # Z 方向的外部力

  # 邊界條件設定
  # type: 0 = 週期性邊界 (periodic), 1 = 固定壓力邊界 (fix pressure), 2 = 固定速度邊界 (fix velocity)
  bc_x_left:
    type: 2
    rho: 1.0                          # 邊界壓力值 (僅在 type=1 時有效)
    vx: 1.0e-2                        # 邊界 X 方向速度 (僅在 type=2 時有效)
    vy: 0.0                           # 邊界 Y 方向速度 (僅在 type=2 時有效)
    vz: 0.0                           # 邊界 Z 方向速度 (僅在 type=2 時有效)
  bc_x_right:
    type: 1
    rho: 1.0
    vx: 0.0
    vy: 0.0
    vz: 0.0
  bc_y_left:
    type: 0
    rho: 1.0
    vx: 0.0
    vy: 0.0
    vz: 0.0
  bc_y_right:
    type: 0
    rho: 1.0
    vx: 0.0
    vy: 0.0
    vz: 0.0
  bc_z_left:
    type: 0
    rho: 1.0
    vx: 0.0
    vy: 0.0
    vz: 0.0
  bc_z_right:
    type: 0
    rho: 1.0
    vx: 0.0
    vy: 0.0
    vz: 0.0

  # 模擬與輸出設定
  max_timestep: 50000                 # 最大模擬時間步長
  output_frequency: 1000              # 終端輸出的頻率
  vtk_frequency: 10000                # VTK 檔案輸出的頻率
```

## 案例資料夾結構

當您執行 `run.py` 腳本時，它將會根據 `config.yml` 中的 `case_name` 參數，在專案根目錄下的 `cases/` 資料夾中建立一個新的子資料夾 (例如 `cases/my_first_case`)。所有與該次模擬相關的輸入檔案 (如體素化後的幾何檔案 `geometry.txt`) 和輸出檔案 (如 `.vtk` 結果檔案) 都將會儲存在這個案例資料夾中。

這樣，您可以輕鬆地管理多個不同的模擬案例，而不會混淆檔案。

## 階段式執行

透過 `config.yml` 中的 `run_mesh` 和 `run_simulation` 旗標，您可以控制 pipeline 的執行階段：

*   `run_mesh: true`：將執行 `voxelize_geometry.py` 來生成體素化幾何。如果您的幾何已經準備好，可以將其設定為 `false` 來跳過此步驟。
*   `run_simulation: true`：將執行 `lbm_solver_3d.py` 來進行 LBM 模擬。如果您只想準備幾何而不執行模擬，可以將其設定為 `false`。
