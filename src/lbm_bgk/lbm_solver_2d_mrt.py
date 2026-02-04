# 基於 Taichi 語言的 Lattice Boltzmann Method (LBM) 流體解算器
# 版本: MRT (Multi-Relaxation Time) - 高穩定性版本
# 作者 : Wang (hietwll@gmail.com)
# 整理與註解 : Gemini

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
        cy=0,  # 是否放置圓柱障礙物 (1: 是, 0: 否)
        cy_para=[0.0, 0.0, 0.0],  # 圓柱參數: [圓心x, 圓心y, 半徑]
    ):
        self.name = name
        self.nx = nx  # LBM 慣例: dx = dy = dt = 1.0 (格子單位)
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
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))  # 速度場 (二維向量)
        self.mask = ti.field(float, shape=(nx, ny))  # 固體遮罩 (1.0: 障礙物, 0.0: 流體)

        # 微觀分佈函數 (Distribution Functions)
        # f_old: 串流前的分佈, f_new: 碰撞後的分佈
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))

        # 邊界條件參數場
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        # 圓柱參數
        self.cy = cy
        self.cy_para = tm.vec3(cy_para)

        # --- D2Q9 模型常數 ---
        # 權重 w (中心: 4/9, 軸向: 1/9, 對角: 1/36)
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

        # 離散速度向量 e (c_i)
        # 順序: [0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1]
        self.e = ti.types.matrix(9, 2, int)(
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
        )

        # ## --- MRT (多鬆弛時間) 核心矩陣與參數 --- ##

        # 1. 轉換矩陣 M (由 Lallemand & Luo 提出)
        # 用於將 f (粒子空間) 轉換為 m (矩空間)
        # 矩 m 的物理意義對應: [rho, e, eps, jx, qx, jy, qy, pxx, pxy]
        self.M = ti.types.matrix(9, 9, float)(
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # rho (密度)
            [-4, -1, -1, -1, -1, 2, 2, 2, 2],  # e (能量)
            [4, -2, -2, -2, -2, 1, 1, 1, 1],  # eps (能量平方)
            [0, 1, 0, -1, 0, 1, -1, -1, 1],  # jx (x動量)
            [0, -2, 0, 2, 0, 1, -1, -1, 1],  # qx (x熱流)
            [0, 0, 1, 0, -1, 1, 1, -1, -1],  # jy (y動量)
            [0, 0, -2, 0, 2, 1, 1, -1, -1],  # qy (y熱流)
            [0, 1, -1, 1, -1, 0, 0, 0, 0],  # pxx (對角應力)
            [0, 0, 0, 0, 0, 1, -1, 1, -1],  # pxy (切應力)
        )

        # 2. 逆矩陣 M_inv (手動定義以提高精度與速度)
        # 用於將 m (矩空間) 轉回 f (粒子空間)
        self.M_inv = (
            ti.types.matrix(9, 9, float)(
                [4, -4, 4, 0, 0, 0, 0, 0, 0],
                [4, -1, -2, 6, -6, 0, 0, 9, 0],
                [4, -1, -2, 0, 0, 6, -6, -9, 0],
                [4, -1, -2, -6, 6, 0, 0, 9, 0],
                [4, -1, -2, 0, 0, -6, 6, -9, 0],
                [4, 2, 1, 6, 3, 6, 3, 0, 9],
                [4, 2, 1, -6, -3, 6, 3, 0, -9],
                [4, 2, 1, -6, -3, -6, -3, 0, 9],
                [4, 2, 1, 6, 3, -6, -3, 0, -9],
            )
            / 36.0
        )  # 歸一化係數

        # 3. 鬆弛矩陣對角線 S (Relaxation Parameters)
        # MRT 的優勢：可以針對不同的物理量設定不同的鬆弛速度
        self.s_nu = self.inv_tau  # 物理相關鬆弛率：控制黏滯性 (針對應力矩 pxx, pxy)
        self.s_other = 1.1  # 數值穩定鬆弛率：用於抑制高階非物理震盪 (建議 1.0 ~ 1.2)

        # S 向量對應 M 矩陣的列
        # 0, 3, 5 是守恆量 (質量, 動量)，碰撞前後不變，設為 0 或任意值皆可
        self.S = ti.types.vector(9, float)(
            0.0,  # 0: rho
            self.s_other,  # 1: e
            self.s_other,  # 2: eps
            0.0,  # 3: jx
            self.s_other,  # 4: qx
            0.0,  # 5: jy
            self.s_other,  # 6: qy
            self.s_nu,  # 7: pxx (決定黏度)
            self.s_nu,  # 8: pxy (決定黏度)
        )

    # ## --- 輔助函數 --- ##

    @ti.func
    def f_eq(self, i, j):
        """
        計算 BGK 模型的平衡分佈函數 (用於邊界條件初始化)
        公式: f_eq = w * rho * [1 + 3(e.u) + 4.5(e.u)^2 - 1.5(u.u)]
        """
        eu = self.e @ self.vel[i, j]  # e . u
        uv = tm.dot(self.vel[i, j], self.vel[i, j])  # u . u
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.func
    def get_meq(self, rho, u, v):
        """
        計算 MRT 模型的平衡矩 (Equilibrium Moments)
        這是在矩空間進行碰撞的核心依據
        """
        u2 = u * u + v * v

        # 根據 D2Q9 模型推導的平衡矩公式
        m0 = rho  # density
        m1 = rho * (-2.0 + 3.0 * u2)  # energy
        m2 = rho * (1.0 - 3.0 * u2)  # energy square
        m3 = rho * u  # momentum-x
        m4 = -rho * u  # heat flux-x
        m5 = rho * v  # momentum-y
        m6 = -rho * v  # heat flux-y
        m7 = rho * (u * u - v * v)  # stress-xx
        m8 = rho * u * v  # stress-xy

        return ti.types.vector(9, float)(m0, m1, m2, m3, m4, m5, m6, m7, m8)

    # ## --- 核心 Kernel 函數 --- ##

    @ti.kernel
    def init(self):
        """系統初始化"""
        self.vel.fill(0)
        self.rho.fill(1)
        self.mask.fill(0)

        # 初始化所有網格
        for i, j in self.rho:
            # 初始狀態設為平衡態，避免初始震盪
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

            # 設定圓柱障礙物 Mask
            if self.cy == 1:
                # 判斷點是否在圓內: (x-x0)^2 + (y-y0)^2 <= r^2
                dist_sq = (i - self.cy_para[0]) ** 2 + (j - self.cy_para[1]) ** 2
                if dist_sq <= self.cy_para[2] ** 2:
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self):
        """MRT 碰撞與串流步 (核心算法)"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            # --- 1. 串流 (Streaming) ---
            # 從鄰居格子抓取粒子 (Pull scheme)
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                f_temp[k] = self.f_old[ip, jp][k]

            # --- 2. 空間轉換 (Transform f -> m) ---
            # 將分佈函數轉換到矩空間
            m = self.M @ f_temp

            # --- 3. 碰撞 (Collision in Moment Space) ---
            # 3.1 計算當地的巨觀量 (使用串流後的數據)
            rho_local = m[0]
            u_local = 0.0
            v_local = 0.0
            if rho_local > 0:
                u_local = m[3] / rho_local
                v_local = m[5] / rho_local

            # 3.2 計算平衡矩 m_eq
            m_eq = self.get_meq(rho_local, u_local, v_local)

            # 3.3 執行鬆弛 m* = m - S(m - m_eq)
            # 這一步讓每個物理量以不同的速度回歸平衡
            m_star = m - self.S * (m - m_eq)

            # --- 4. 逆轉換 (Transform m -> f) ---
            # 將碰撞後的矩轉回分佈函數，並寫入下一步
            self.f_new[i, j] = self.M_inv @ m_star

    @ti.kernel
    def update_macro_var(self):
        """更新巨觀變量 (密度與速度) 並同步數據"""
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            local_rho = 0.0
            local_vel = tm.vec2(0.0, 0.0)

            for k in ti.static(range(9)):
                # 1. Swap: 將 f_new (下一步) 搬回 f_old (當前步)
                # 這是為了讓下一個時間步的 Streaming 能讀取到正確數據
                self.f_old[i, j][k] = self.f_new[i, j][k]

                # 2. 統計密度 (0階矩)
                local_rho += self.f_new[i, j][k]

                # 3. 統計動量 (1階矩)
                local_vel += tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]

            # 4. 更新全域變數
            self.rho[i, j] = local_rho
            if local_rho > 0:
                self.vel[i, j] = local_vel / local_rho
            else:
                self.vel[i, j] = tm.vec2(0.0, 0.0)

    @ti.kernel
    def apply_bc(self):
        """應用邊界條件"""

        # --- A. 四周邊框處理 (Box Boundaries) ---
        # 使用 NEEM (非平衡態外推) 以獲得高精度

        # 1. 左右邊界 (j: 1 ~ ny-1)
        for j in range(1, self.ny - 1):
            # 左邊界 (Inlet/Wall)
            self.apply_bc_core(1, 0, 0, j, 1, j)
            # 右邊界 (Outlet/Wall)
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # 2. 上下邊界 (i: 0 ~ nx)
        for i in range(self.nx):
            # 上邊界
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)
            # 下邊界
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # --- B. 障礙物處理 (Cylinder) ---
        # 使用 "強制平衡法" 以獲得最大穩定性 (避免梯度爆炸)

        if self.cy == 1:
            for i, j in ti.ndrange(self.nx, self.ny):
                if self.mask[i, j] == 1:
                    # 1. 強制速度歸零 (No-slip)
                    self.vel[i, j] = 0.0, 0.0

                    # 2. 使用 "強制平衡態" 重設分佈函數
                    # f = f_eq(rho=1, u=0)，像海綿一樣吸收衝擊
                    # 這裡為了簡單，假設內部密度不變或設為 1.0
                    self.f_old[i, j] = self.f_eq(i, j)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        """
        通用邊界處理核心函數
        方法: Non-Equilibrium Extrapolation Method (NEEM)
        """
        if outer == 1:
            if self.bc_type[dr] == 0:
                # Dirichlet: 強制速度 (如入口)
                self.vel[ibc, jbc] = self.bc_value[dr]
            elif self.bc_type[dr] == 1:
                # Neumann: 速度梯度為0 (如出口)，跟鄰居一樣
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        # 壓力/密度邊界條件：設為鄰居密度
        self.rho[ibc, jbc] = self.rho[inb, jnb]

        # NEEM 核心公式：f_boundary = f_eq_bc + (f_neighbor - f_eq_neighbor)
        # 意義：保留鄰居的非平衡擾動，疊加到邊界的平衡態上
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    # ## --- 主程式與視覺化 --- ##

    def solve(self):
        """主解算迴圈"""
        print(f"Starting simulation: {self.name}")
        print(
            f"Grid: {self.nx}x{self.ny}, Viscosity: {self.niu}, Relaxation Tau: {self.tau:.4f}"
        )

        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()

        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # 每幀執行 10 次模擬步
            for _ in range(10):
                self.collide_and_stream()  # MRT 碰撞
                self.update_macro_var()  # 更新巨觀量
                self.apply_bc()  # 邊界修正

            # --- 視覺化數據準備 ---
            vel = self.vel.to_numpy()

            # 計算渦度 (Vorticity)
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]

            # 計算速度大小
            vel_mag = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)

            # --- 繪圖 ---
            # 1. 渦度圖 (上)
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
            vor_img = cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
            ).to_rgba(vor)

            # 2. 速度場圖 (下)
            vel_img = cm.plasma(vel_mag / 0.15)

            # 拼接並顯示
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()


# --- 程式入口點 ---

if __name__ == "__main__":
    flow_case = 0 if len(sys.argv) < 2 else int(sys.argv[1])

    if flow_case == 0:
        # Case 0: 卡門渦街 (Von Karman Vortex Street)
        lbm = lbm_solver(
            name="Karman Vortex Street (MRT)",
            nx=801,
            ny=201,
            niu=0.0005,  # 低黏滯性，MRT 應該能穩定運行
            bc_type=[0, 0, 1, 0],  # 左入, 上壁, 右出, 下壁
            bc_value=[[0.05, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            cy=1,
            cy_para=[160.0, 100.0, 20.0],
        )
        lbm.solve()

    elif flow_case == 1:
        # Case 1: 頂蓋驅動穴流 (Lid-driven Cavity)
        lbm = lbm_solver(
            name="Lid-driven Cavity (MRT)",
            nx=256,
            ny=256,
            niu=0.01,
            bc_type=[0, 0, 0, 0],
            bc_value=[[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]],  # 僅上方有速度
            cy=0,
        )
        lbm.solve()

    else:
        print("Invalid case ID.")
