import matplotlib
import numpy as np
from matplotlib import cm
import taichi as ti
import taichi.math as tm

from VideoRecorder import VideoRecorder
from scipy.ndimage import gaussian_filter

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

ti.init(arch=ti.gpu)


@ti.data_oriented
class LBM2D_MRT_LES:
    # ------------------------------------------------
    # region init
    def __init__(self, config, mask_data=None):
        """
        LBM Solver 初始化入口
        """
        self.config = config

        # 1. 讀取模擬與物理參數
        self._init_params()

        # 2. 配置 Taichi 記憶體 (Fields)
        self._init_fields(mask_data)

        # 3. 定義 LBM/MRT 常數與矩陣 (優化重點)
        self._init_constants()

        # # 4. 初始化場數值 (由 0 開始或讀取設定)
        # self.init_sim()

    #  init 子函式: 參數讀取
    def _init_params(self):
        sim_cfg = self.config["simulation"]

        # 基礎幾何與時間
        self.name = sim_cfg["name"]
        self.nx = sim_cfg["nx"]
        self.ny = sim_cfg["ny"]
        self.steps_per_frame = sim_cfg.get("steps_per_frame", 10)
        self.warmup_steps = sim_cfg.get("warmup_steps", 0)

        # 物理參數
        self.niu = sim_cfg["niu"]
        self.tau_0 = 3.0 * self.niu + 0.5

        # LES (大渦模擬) 參數
        self.C_smag = sim_cfg.get("smagorinsky_constant", 0.15)
        self.Cs_sq_factor = 18.0 * (self.C_smag**2)

        # MRT 鬆弛參數
        self.S_other = sim_cfg.get("ghost_moments_s", 1.2)

        # 視覺化參數
        self.viz_sigma = sim_cfg.get("visualization_gaussian_sigma", 1.0)

    #  init 子函式: 記憶體配置 (Fields)
    def _init_fields(self, mask_data):
        # 巨觀量 (Macro-scopic)
        self.rho = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.nx, self.ny))

        # 分布函數 (Micro-scopic, f_old / f_new)
        self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(self.nx, self.ny))

        # 遮罩 (Mask)
        self.mask = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        if mask_data is not None:
            self.mask.from_numpy(mask_data.astype(np.float32))
        else:
            self.mask.fill(0.0)

        # 邊界條件 (Boundary Conditions)
        bc_cfg = self.config["boundary_condition"]
        self.bc_type = ti.field(dtype=ti.i32, shape=4)
        self.bc_value = ti.Vector.field(2, dtype=ti.f32, shape=4)

        self.bc_type.from_numpy(np.array(bc_cfg["type"], dtype=np.int32))
        self.bc_value.from_numpy(np.array(bc_cfg["value"], dtype=np.float32))

        # 統計與計數
        self.frame_count = ti.field(dtype=ti.i32, shape=())

    #  init 子函式: 常數與矩陣 (Constants)
    def _init_constants(self):
        # D2Q9 權重
        self.w = ti.types.vector(9, ti.f32)(
            4.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 9.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
            1.0 / 36.0,
        )

        # D2Q9 離散速度向量
        self.e = ti.types.matrix(9, 2, ti.i32)(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
                [1, 1],
                [-1, 1],
                [-1, -1],
                [1, -1],
            ]
        )

        # --- MRT 轉換矩陣 (核心優化) ---
        # 這裡將 M 定義為 Taichi Matrix 而非 Field，
        M_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                [4, -2, -2, -2, -2, 1, 1, 1, 1],
                [0, 1, 0, -1, 0, 1, -1, -1, 1],
                [0, -2, 0, 2, 0, 1, -1, -1, 1],
                [0, 0, 1, 0, -1, 1, 1, -1, -1],
                [0, 0, -2, 0, 2, 1, 1, -1, -1],
                [0, 1, -1, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 1, -1],
            ],
            dtype=np.float32,
        )

        self.M = ti.Matrix(M_np)
        self.invM = ti.Matrix(np.linalg.inv(M_np))

        # MRT 鬆弛對角矩陣 (S vector)
        # 用於碰撞步驟: m* = m - S * (m - m_eq)
        # S_other 是 Ghost Moments 的鬆弛率
        self.S_base = ti.types.vector(9, ti.f32)(
            0.0,  # density (conserved)
            self.S_other,  # energy
            self.S_other,  # epsilon
            0.0,  # jx (conserved)
            self.S_other,  # qx
            0.0,  # jy (conserved)
            self.S_other,  # qy
            0.0,  # pxx (conserved in standard LBM, but relaxed here)
            0.0,  # pxy
        )

    # endregion

    # ------------------------------------------------
    # region LBM MRT-LES Kernels and Functions
    def get_physical_fields(self):
        """
        將 GPU 上的速度場與遮罩數據導出為 NumPy 數組
        """
        # .to_numpy() 會自動處理數據同步與拷貝
        return self.vel.to_numpy(), self.mask.to_numpy()

    @ti.func
    def f_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func
    def get_meq(self, rho, u, v):
        u2 = u * u + v * v
        return ti.types.vector(9, float)(
            rho,
            rho * (-2.0 + 3.0 * u2),
            rho * (1.0 - 3.0 * u2),
            rho * u,
            -rho * u,
            rho * v,
            -rho * v,
            rho * (u * u - v * v),
            rho * u * v,
        )

    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        self.frame_count[None] = 0
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # --- Stream ---
            f_temp = ti.Vector([0.0] * 9, dt=ti.f32)
            for k in ti.static(range(9)):
                ip = (i - self.e[k, 0] + self.nx) % self.nx
                jp = (j - self.e[k, 1] + self.ny) % self.ny
                f_temp[k] = self.f_old[ip, jp][k]

            # --- Collision (MRT) ---
            # 直接使用 ti.Matrix @ ti.Vector
            m = self.M @ f_temp
            rho_l = m[0]
            u_l = m[3] / rho_l
            v_l = m[5] / rho_l

            m_eq = self.get_meq(rho_l, u_l, v_l)

            # LES 渦黏計算 (計算應變率張量相關項)
            # Sij 相關於非平衡動態矩 m[7], m[8]
            neq_7 = m[7] - m_eq[7]
            neq_8 = m[8] - m_eq[8]
            S_mag = tm.sqrt(neq_7**2 + neq_8**2)

            tau_eff = self.tau_0
            if ti.static(self.C_smag > 0):
                # Smagorinsky 模型修正
                tau_eddy = 0.5 * (
                    tm.sqrt(self.tau_0**2 + 2 * self.Cs_sq_factor * S_mag) - self.tau_0
                )
                tau_eff += tau_eddy

            s_eff = 1.0 / tau_eff
            S_local = self.S_base
            S_local[7] = S_local[8] = s_eff  # 鬆弛剪切應力項

            m_star = m - S_local * (m - m_eq)
            self.f_new[i, j] = self.invM @ m_star

    @ti.kernel
    def update_macro_var(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            local_rho = 0.0
            local_vel = tm.vec2(0.0, 0.0)
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                local_rho += self.f_new[i, j][k]
                local_vel += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            self.rho[i, j] = local_rho
            if local_rho > 0:
                self.vel[i, j] = local_vel / local_rho
            else:
                self.vel[i, j] = tm.vec2(0, 0)

    @ti.kernel
    def apply_bc(self):
        self.frame_count[None] += 1
        # 緩啟動
        ramp = tm.min(1.0, float(self.frame_count[None]) / self.warmup_steps)

        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j, ramp)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j, ramp)
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2, ramp)
            self.apply_bc_core(1, 3, i, 0, i, 1, ramp)

        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.vel[i, j] = 0.0, 0.0
                self.f_old[i, j] = self.f_eq(i, j)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb, ramp: float):
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr] * ramp
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    def check_re(self):
        u_vec = self.bc_value[0]
        u_char = np.sqrt(u_vec[0] ** 2 + u_vec[1] ** 2)
        # 嘗試從 config 讀取特徵長度 (CL), 若無則用預設值
        l_char = self.config["boundary_condition"].get("CL", 20.0)
        if self.config["mask"]["type"] == "cylinder":
            l_char = self.config["mask"]["params"]["r"] * 2

        print(f"--- [LES Info] ---")
        print(f"Smagorinsky Constant (Cs): {self.C_smag}")
        print(f"Ghost Moments S: {self.S_other} (Read from Config)")
        utils.print_reynolds_info(u_char, l_char, self.niu, "Characteristic Length")
        return (u_char * l_char) / self.niu

    def run_step(self, steps=1):
        """
        [核心運算層]
        執行 LBM 的物理時間步推進。

        標準循環：
        1. Collide & Stream: 計算分佈函數的碰撞與流動 (f_old -> f_new)
        2. Update Macro: 計算密度與速度，並交換緩衝區 (f_new -> f_old)
        3. Apply BC: 強制設定邊界條件與障礙物處理
        """
        for _ in range(steps):
            # 1. 碰撞與串流 (計算 f_new)
            self.collide_and_stream()

            # 2. 更新巨觀量 (rho, vel) 並將 f_new 寫回 f_old
            # 注意：你的 update_macro_var kernel 內包含了 f_old[i,j] = f_new[i,j]
            # 這一步至關重要，否則下一幀計算會用到舊數據
            self.update_macro_var()

            # 3. 應用邊界條件 (Inlet/Outlet) 與 障礙物 (Mask)
            # 這會覆蓋邊界上的 vel 和 rho，確保物理場正確
            self.apply_bc()

    # endregion
    # ------------------------------------------------

    # ------------------------------------------------
    # region Visualization Helpers (新增的渲染輔助區塊)

    def _init_render_resources(self):
        """初始化繪圖資源 (Colormap)"""
        colors = [
            (1, 1, 0),
            (0.953, 0.490, 0.016),
            (0, 0, 0),
            (0.176, 0.976, 0.529),
            (0, 1, 1),
        ]
        self.my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "my_cmap", colors
        )
        self.my_cmap.set_bad(color="grey")

        # 預先定義 Normalizer，避免每幀重新建立
        self.vor_norm = matplotlib.colors.Normalize(vmin=-0.03, vmax=0.03)

    def _render_frame(self, vel_raw, mask_np):
        """
        [渲染核心] 輸入原始速度與遮罩，輸出組合好的 RGB 圖像
        """
        # 1. 數據後處理 (Post-processing)
        vel_x = gaussian_filter(vel_raw[:, :, 0], sigma=self.viz_sigma)
        vel_y = gaussian_filter(vel_raw[:, :, 1], sigma=self.viz_sigma)

        # 計算渦度 (Vorticity)
        ugrad = np.gradient(vel_x)
        vgrad = np.gradient(vel_y)
        vor = ugrad[1] - vgrad[0]
        vel_mag = np.sqrt(vel_x**2 + vel_y**2)

        # 處理遮罩 (Masking)
        vor[mask_np > 0] = np.nan

        # 2. 轉換為影像 (Scalar Mapping)
        # Vorticity Image
        vor_mapper = cm.ScalarMappable(norm=self.vor_norm, cmap=self.my_cmap)
        vor_img = vor_mapper.to_rgba(vor)[:, :, :3]

        # Velocity Image (Plasma colormap)
        vel_img = cm.plasma(vel_mag / 0.15)[:, :, :3]

        # 3. 疊加障礙物顏色 (灰色)
        # 利用廣播機制加速，取代原本的 for loop
        mask_indices = mask_np == 1
        gray_color = 0.5
        vor_img[mask_indices] = gray_color
        vel_img[mask_indices] = gray_color

        # 4. 拼接影像 (左右拼接)
        # 輸出 Shape: (nx, ny*2, 3)
        return np.concatenate((vel_img, vor_img), axis=1)

    # endregion

    # ------------------------------------------------
    # region main solver (重構後的 solve)
    def solve(self):
        """
        主執行函式：負責初始化視窗、錄影與主迴圈控制
        """
        # 1. 初始化模擬與資源
        self.init()
        self._init_render_resources()
        self.Re = self.check_re()

        # 2. 初始化 GUI
        # 畫面高度為 ny * 2 (因為是速度圖+渦度圖拼接)
        gui = ti.GUI(self.name, (self.nx, self.ny * 2))

        # 3. 初始化錄影 (路徑容錯處理)
        try:
            # 嘗試適配不同的 config 結構
            if "foler_paths" in self.config and "output" in self.config["foler_paths"]:
                out_dir = self.config["foler_paths"]["output"]
            else:
                out_dir = self.config.get("output", {}).get("video_dir", "./output")

            # 確保目錄存在
            os.makedirs(out_dir, exist_ok=True)
            video_path = os.path.join(out_dir, f"Re{int(self.Re)}_nx{self.nx}.mp4")

            recorder = VideoRecorder(video_path, self.nx, self.ny * 2, fps=30)
            recorder.start()
        except Exception as e:
            print(f"[Warning] VideoRecorder init failed: {e}")
            recorder = None

        print(f"--- Simulation Started: Re={self.Re:.2f} ---")

        try:
            while gui.running:
                # --- A. 物理計算 ---
                # 呼叫既有的 run_step (包含 collide, stream, bc, macro_update)
                self.run_step(self.steps_per_frame)

                # --- B. 獲取數據 ---
                vel_raw, mask_np = self.get_physical_fields()

                # --- C. 渲染處理 (封裝在 _render_frame) ---
                img_gui = self._render_frame(vel_raw, mask_np)

                # --- D. 顯示與錄影 ---
                gui.set_image(img_gui)
                gui.show()

                if recorder:
                    # GUI (W, H) -> Video (H, W) 轉置處理
                    img_video = np.transpose(img_gui, (1, 0, 2))
                    recorder.write_frame(img_video)

        except Exception as e:
            print(f"Simulation crashed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if recorder:
                recorder.stop()
            print("Simulation finished.")

    # endregion
