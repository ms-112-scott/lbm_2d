# 基於 Taichi 語言的 Lattice Boltzmann Method (LBM) 流體解算器
# 版本: Configurable MRT + Utils Refactor
# 運行: python src/lbm_bgk/lbm2d_mrt_config.py

import matplotlib
import numpy as np
from matplotlib import cm
import taichi as ti
import taichi.math as tm
import sys
import os

# 動態加入路徑以匯入 utils (如果執行目錄不同)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 匯入自定義工具模組
import utils

# 初始化 Taichi，優先使用 GPU
ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver:
    def __init__(self, config, mask_data=None):
        self.config = config
        sim_cfg = config["simulation"]
        bc_cfg = config["boundary_condition"]

        self.name = sim_cfg["name"]
        self.nx = sim_cfg["nx"]
        self.ny = sim_cfg["ny"]
        self.niu = sim_cfg["niu"]
        self.steps_per_frame = sim_cfg.get("steps_per_frame", 10)

        self.bc_type_list = bc_cfg["type"]
        self.bc_value_list = bc_cfg["value"]

        # --- 物理參數 ---
        self.tau = 3.0 * self.niu + 0.5
        self.inv_tau = 1.0 / self.tau

        # --- Taichi 場定義 ---
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.mask = ti.field(float, shape=(self.nx, self.ny))

        # 載入 Mask
        if mask_data is not None:
            if mask_data.shape != (self.nx, self.ny):
                raise ValueError(
                    f"Mask shape mismatch: {mask_data.shape} != ({self.nx}, {self.ny})"
                )
            self.mask.from_numpy(mask_data.astype(np.float32))
        else:
            self.mask.fill(0.0)

        # 分佈函數
        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))

        # 邊界條件場
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(self.bc_type_list, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(self.bc_value_list, dtype=np.float32))

        # --- D2Q9 常數 ---
        self.w = ti.types.vector(9, float)(
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
        self.e = ti.types.matrix(9, 2, int)(
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

        # --- MRT 矩陣 ---
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

        M_inv_np = np.linalg.inv(M_np)  # 讓 Numpy 幫忙算逆矩陣比較準確且乾淨

        self.M_field = ti.field(float, shape=(9, 9))
        self.M_inv_field = ti.field(float, shape=(9, 9))
        self.M_field.from_numpy(M_np)
        self.M_inv_field.from_numpy(M_inv_np)

        self.s_nu = self.inv_tau
        self.s_other = 1.1  # 可調整的鬆弛參數
        self.S = ti.types.vector(9, float)(
            0,
            self.s_other,
            self.s_other,
            0,
            self.s_other,
            0,
            self.s_other,
            self.s_nu,
            self.s_nu,
        )

    # --- 輔助函數 ---
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

    # --- 核心 Kernel ---
    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(1)
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collide_and_stream(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # 1. Stream
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                f_temp[k] = self.f_old[ip, jp][k]

            # 2. Moment Transform
            # --- [修正] 直接初始化為向量，刪除錯誤的 scalar 定義 ---
            m = ti.types.vector(9, float)(0.0)

            # 矩陣乘法: m = M * f_temp
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_field[r, c] * f_temp[c]
                m[r] = val

            # 3. Collide
            rho_l = m[0]
            u_l, v_l = 0.0, 0.0
            if rho_l > 0:
                u_l, v_l = m[3] / rho_l, m[5] / rho_l

            m_eq = self.get_meq(rho_l, u_l, v_l)
            m_star = m - self.S * (m - m_eq)

            # 4. Inverse Transform
            f_new_val = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_inv_field[r, c] * m_star[c]
                f_new_val[r] = val

            self.f_new[i, j] = f_new_val

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
        # Left/Right
        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)
        # Top/Bottom
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # Cylinder Mask BC
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.vel[i, j] = 0.0, 0.0
                self.f_old[i, j] = self.f_eq(i, j)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    # --- 整合 utils 的 Re 計算 ---
    def check_re(self):
        # 準備參數
        u_vec = self.bc_value_list[0]
        u_char = np.sqrt(u_vec[0] ** 2 + u_vec[1] ** 2)

        # 嘗試取得特徵長度
        try:
            r = self.config["mask"]["params"]["r"]
            l_char = 2.0 * r
            name = "Cylinder Diameter"
        except:
            l_char = self.ny
            name = "Channel Height"

        # 呼叫 utils 函數
        utils.print_reynolds_info(u_char, l_char, self.niu, name)

    # ## --- 主程式與視覺化 --- ##
    def solve(self):
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()

        print(f"Simulation started. Grid: {self.nx}x{self.ny}")

        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # 每偵計算 10 步
            for _ in range(self.steps_per_frame):
                self.collide_and_stream()
                self.update_macro_var()
                self.apply_bc()

            # --- 視覺化 ---
            vel = self.vel.to_numpy()
            # 計算渦度
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)

            # 處理 Mask 顯示
            mask_np = self.mask.to_numpy()
            vor[mask_np > 0] = np.nan  # 讓障礙物在渦度圖上“挖空”

            # 設定 Colormap
            colors = [
                (1, 1, 0),
                (0.953, 0.490, 0.016),
                (0, 0, 0),
                (0.176, 0.976, 0.529),
                (0, 1, 1),
            ]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                "my_cmap", colors
            )
            my_cmap.set_bad(color="grey")  # NaN 的地方顯示灰色

            vor_img = cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
            ).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)

            # 確保障礙物在速度圖也顯示為灰色
            for d in range(3):
                vel_img[:, :, d] = np.where(mask_np == 1, 0.5, vel_img[:, :, d])

            # 拼接並顯示
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()


# --- Entry Point ---
if __name__ == "__main__":
    # 使用 utils 載入設定
    config_path = "src/lbm_bgk/config2.yaml"  # 請確認路徑是否正確
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = utils.load_config(config_path)

    # 使用 utils 產生 Mask
    nx, ny = config["simulation"]["nx"], config["simulation"]["ny"]
    mask_cfg = config.get("mask", {})
    mask = None

    if mask_cfg.get("enable"):
        print(f"Generating Mask: {mask_cfg['type']}")
        if mask_cfg["type"] == "cylinder":
            p = mask_cfg["params"]
            mask = utils.create_cylinder_mask(nx, ny, p["cx"], p["cy"], p["r"])

    solver = lbm_solver(config, mask_data=mask)
    solver.solve()
