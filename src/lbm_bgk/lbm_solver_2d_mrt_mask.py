# 基於 Taichi 語言的 Lattice Boltzmann Method (LBM) 流體解算器
# 版本: MRT (Multi-Relaxation Time) - 高穩定性版本
# 修改內容: 修復 9x9 Matrix Warning，整合外部 Mask 輸入
# 作者 : Wang (hietwll@gmail.com)
# 重構與註解 : Gemini

import sys
import matplotlib
import numpy as np
from matplotlib import cm
import taichi as ti
import taichi.math as tm

# 初始化 Taichi，優先使用 GPU 進行平行運算加速
ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name,  # 模擬案例名稱
        nx,  # 計算域寬度 (x方向網格數)
        ny,  # 計算域高度 (y方向網格數)
        niu,  # 流體運動黏滯係數 (Kinematic Viscosity)
        bc_type,  # 邊界條件類型 [左, 上, 右, 下]: 0 -> Dirichlet (固定速度); 1 -> Neumann (自由流出)
        bc_value,  # 邊界條件數值: 如果 bc_type = 0，這裡指定速度向量 [u, v]
        mask_data=None,  # [新增] 外部傳入的障礙物遮罩 (Numpy Array)
    ):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.niu = niu

        # --- 物理參數計算 ---
        # 計算鬆弛時間 (Relaxation time) tau
        # 公式: niu = (cs^2) * (tau - 0.5)，其中聲速 cs^2 = 1/3
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        # --- Taichi 場定義 (Fields) ---
        # 巨觀量場
        self.rho = ti.field(float, shape=(nx, ny))  # 密度場
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))  # 速度場
        self.mask = ti.field(float, shape=(nx, ny))  # 固體遮罩

        # [新增] 載入 Mask 數據
        if mask_data is not None:
            if mask_data.shape != (nx, ny):
                raise ValueError("Mask shape must match grid size (nx, ny)")
            self.mask.from_numpy(mask_data.astype(np.float32))
        else:
            self.mask.fill(0.0)  # 0.0 代表流體，1.0 代表固體

        # 微觀分佈函數 (Distribution Functions)
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))

        # 邊界條件參數場
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        # --- D2Q9 模型常數 ---

        # 權重 w
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

        # 離散速度向量 e
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

        # ## --- MRT 核心矩陣 (修正 Warning 版) --- ##
        # 原本使用 ti.types.matrix 定義 9x9 會導致編譯器嘗試將其塞入暫存器 (Registers)
        # 超過 32 個變數會導致效能下降與編譯警告。
        # 解決方案：將矩陣存入 ti.field (Global Memory)，雖然讀取稍慢一點，但編譯快且穩定。

        # 1. 定義轉換矩陣 M (Numpy 端)
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

        # 2. 定義逆矩陣 M_inv (Numpy 端)
        M_inv_np = (
            np.array(
                [
                    [4, -4, 4, 0, 0, 0, 0, 0, 0],
                    [4, -1, -2, 6, -6, 0, 0, 9, 0],
                    [4, -1, -2, 0, 0, 6, -6, -9, 0],
                    [4, -1, -2, -6, 6, 0, 0, 9, 0],
                    [4, -1, -2, 0, 0, -6, 6, -9, 0],
                    [4, 2, 1, 6, 3, 6, 3, 0, 9],
                    [4, 2, 1, -6, -3, 6, 3, 0, -9],
                    [4, 2, 1, -6, -3, -6, -3, 0, 9],
                    [4, 2, 1, 6, 3, -6, -3, 0, -9],
                ],
                dtype=np.float32,
            )
            / 36.0
        )

        # 3. 創建 Taichi Fields 並載入數據
        self.M_field = ti.field(float, shape=(9, 9))
        self.M_inv_field = ti.field(float, shape=(9, 9))
        self.M_field.from_numpy(M_np)
        self.M_inv_field.from_numpy(M_inv_np)

        # 4. 鬆弛參數 S (保持為 Vector，因為 size=9 < 32，這是 OK 的)
        self.s_nu = self.inv_tau
        self.s_other = 1.1
        self.S = ti.types.vector(9, float)(
            0.0,
            self.s_other,
            self.s_other,
            0.0,
            self.s_other,
            0.0,
            self.s_other,
            self.s_nu,
            self.s_nu,
        )

    # ## --- 輔助函數 --- ##

    @ti.func
    def f_eq(self, i, j):
        """計算平衡分佈函數 (BGK 公式)"""
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func
    def get_meq(self, rho, u, v):
        """計算 MRT 平衡矩"""
        u2 = u * u + v * v
        m0 = rho
        m1 = rho * (-2.0 + 3.0 * u2)
        m2 = rho * (1.0 - 3.0 * u2)
        m3 = rho * u
        m4 = -rho * u
        m5 = rho * v
        m6 = -rho * v
        m7 = rho * (u * u - v * v)
        m8 = rho * u * v
        return ti.types.vector(9, float)(m0, m1, m2, m3, m4, m5, m6, m7, m8)

    # ## --- 核心 Kernel --- ##

    @ti.kernel
    def init(self):
        """初始化場變數"""
        self.vel.fill(0)
        self.rho.fill(1)
        # 注意: mask 不在這裡重置，因為已經在 __init__ 載入了
        for i, j in self.rho:
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

    @ti.kernel
    def collide_and_stream(self):
        """MRT 碰撞與串流步 (已修正矩陣乘法)"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # --- 1. 串流 (Streaming) ---
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                f_temp[k] = self.f_old[ip, jp][k]

            # --- 2. 空間轉換 (Transform f -> m) ---
            # [修改] 手動矩陣乘法: m = M_field * f_temp
            m = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                sum_val = 0.0
                for c in ti.static(range(9)):
                    sum_val += self.M_field[r, c] * f_temp[c]
                m[r] = sum_val

            # --- 3. 碰撞 (Collision) ---
            rho_local = m[0]
            u_local = 0.0
            v_local = 0.0
            if rho_local > 0:
                u_local = m[3] / rho_local
                v_local = m[5] / rho_local

            m_eq = self.get_meq(rho_local, u_local, v_local)
            m_star = m - self.S * (m - m_eq)

            # --- 4. 逆轉換 (Transform m -> f) ---
            # [修改] 手動矩陣乘法: f_new = M_inv_field * m_star
            f_new_val = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                sum_val = 0.0
                for c in ti.static(range(9)):
                    sum_val += self.M_inv_field[r, c] * m_star[c]
                f_new_val[r] = sum_val

            self.f_new[i, j] = f_new_val

    @ti.kernel
    def update_macro_var(self):
        """更新巨觀變量"""
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
                self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """應用邊界條件 (包含四周邊框與內部 Mask)"""
        # A. 四周邊框
        for j in range(1, self.ny - 1):
            self.apply_bc_core(1, 0, 0, j, 1, j)  # Left
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)  # Right
        for i in range(self.nx):
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)  # Top
            self.apply_bc_core(1, 3, i, 0, i, 1)  # Bottom

        # B. 任意形狀障礙物處理 (Mask)
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                # 強制速度為 0，且分佈函數重置為靜止平衡態
                self.vel[i, j] = 0.0, 0.0
                self.f_old[i, j] = self.f_eq(i, j)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        """邊界處理核心 (NEEM)"""
        if outer == 1:
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    # ## --- 主程式與視覺化 --- ##
    def solve(self):
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()

        print(f"Simulation started. Grid: {self.nx}x{self.ny}")

        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # 每偵計算 10 步
            for _ in range(10):
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
            print("dwdwd", img.shape)
            gui.set_image(img)
            gui.show()


# --- 輔助 Mask 產生器 ---


def create_cylinder_mask(nx, ny, cx, cy, r):
    """產生圓形 Mask"""
    mask = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            if (i - cx) ** 2 + (j - cy) ** 2 <= r**2:
                mask[i, j] = 1.0
    return mask


# --- 程式入口點 ---
if __name__ == "__main__":
    # 1. 設定解析度
    nx = 801
    ny = 201

    # 2. 產生 Mask (圓柱繞流)
    print("Generating Cylinder Mask...")
    mask = create_cylinder_mask(nx, ny, cx=160, cy=100, r=20)

    # 3. 建立 Solver
    lbm = lbm_solver(
        name="Karman Vortex (External Mask Fixed)",
        nx=nx,
        ny=ny,
        niu=0.0005,
        bc_type=[0, 0, 1, 0],  # 左邊入口固定速度，右邊自由流出
        bc_value=[[0.05, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        mask_data=mask,  # 傳入 Numpy Mask
    )

    # 4. 開始計算
    lbm.solve()
