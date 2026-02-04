import taichi as ti
import taichi.math as tm
import numpy as np
from typing import Dict, Any


@ti.data_oriented
class LbmSolverMrt:
    """
    修正版 LBM Solver MRT
    - 修正鬆弛率矩陣 S，解耦 Ghost Modes
    - 實作直接矩平衡態 (Direct Moment Equilibrium)
    """

    def __init__(self, cfg: Dict[str, Any], mask_data: np.ndarray):
        # 基礎參數設定
        self.nx: int = cfg["simulation"]["nx"]
        self.ny: int = cfg["simulation"]["ny"]
        self.niu_lbm: float = cfg["simulation"]["niu"]

        # --- [FIX 1] 修正鬆弛參數 ---
        # 物理鬆弛率 (控制黏度)
        self.tau = 3.0 * self.niu_lbm + 0.5
        s_nu = 1.0 / self.tau

        # 鬼影模態鬆弛率 (控制穩定性)
        # 這些值通常設為 1.1 ~ 1.2 來抑制高頻噪音，與物理黏度無關
        s_e = 1.1  # Energy
        s_eps = 1.1  # Energy squared
        s_q = 1.1  # Energy flux (Heat flux)

        # MRT 鬆弛向量 S
        # indices: 0:rho, 1:e, 2:eps, 3:jx, 4:qx, 5:jy, 6:qy, 7:pxx, 8:pxy
        # 守恆量 (rho, jx, jy) 設為 0 (因為它們在碰撞中不變，強制設為 m_eq)
        self.S = ti.Vector([0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu], dt=ti.f32)

        inlet_vel: float = cfg["boundaries"]["values"][0][0]
        self.Tc: int = int(self.nx / max(inlet_vel, 0.001))

        # Taichi 場域定義
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.mask = ti.field(float, shape=(self.nx, self.ny))
        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))

        # D2Q9 常數
        self.w = ti.Vector(
            [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
            dt=ti.f32,
        )
        self.e = ti.Matrix(
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
            ],
            dt=ti.i32,
        )

        # MRT 矩陣 M (Lallemand & Luo, 2000)
        self.M = ti.field(ti.f32, shape=(9, 9))
        self.M_inv = ti.field(ti.f32, shape=(9, 9))
        M_np = np.array(
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],  # 0: rho
                [-4, -1, -1, -1, -1, 2, 2, 2, 2],  # 1: e
                [4, -2, -2, -2, -2, 1, 1, 1, 1],  # 2: epsilon
                [0, 1, 0, -1, 0, 1, -1, -1, 1],  # 3: jx
                [0, -2, 0, 2, 0, 1, -1, -1, 1],  # 4: qx
                [0, 0, 1, 0, -1, 1, 1, -1, -1],  # 5: jy
                [0, 0, -2, 0, 2, 1, 1, -1, -1],  # 6: qy
                [0, 1, -1, 1, -1, 0, 0, 0, 0],  # 7: pxx
                [0, 0, 0, 0, 0, 1, -1, 1, -1],  # 8: pxy
            ],
            dtype=np.float32,
        )
        self.M.from_numpy(M_np)
        self.M_inv.from_numpy(np.linalg.inv(M_np))

        # 邊界條件與遮罩
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(
            np.array(cfg["boundaries"]["values"], dtype=np.float32)
        )
        self.mask.from_numpy(mask_data)

    @ti.func
    def get_moments_equilibrium(
        self, rho: float, v: tm.vec2
    ) -> ti.types.vector(9, float):
        """
        [FIX 2] 直接計算矩空間的平衡態 m_eq
        對應 Lallemand & Luo 的 M 矩陣定義
        """
        usq = v.x * v.x + v.y * v.y
        # 使用不可壓縮近似或標準 D2Q9 多項式
        # 這裡使用標準形式

        m_eq = ti.Vector([0.0] * 9, dt=ti.f32)

        # Conserved moments (守恆量)
        m_eq[0] = rho  # rho
        m_eq[3] = rho * v.x  # jx
        m_eq[5] = rho * v.y  # jy

        # Non-conserved moments (非守恆量)
        # e_eq = -2*rho + 3*rho*|u|^2
        m_eq[1] = -2.0 * rho + 3.0 * rho * usq

        # eps_eq = rho - 3*rho*|u|^2
        m_eq[2] = rho - 3.0 * rho * usq

        # qx_eq = -rho * ux
        m_eq[4] = -rho * v.x

        # qy_eq = -rho * uy
        m_eq[6] = -rho * v.y

        # pxx_eq = rho * (ux^2 - uy^2)
        m_eq[7] = rho * (v.x * v.x - v.y * v.y)

        # pxy_eq = rho * ux * uy
        m_eq[8] = rho * v.x * v.y

        return m_eq

    @ti.func
    def mat_vec_mul(
        self, mat: ti.template(), vec: ti.template()
    ) -> ti.types.vector(9, float):
        res = ti.Vector([0.0] * 9, dt=ti.f32)
        for i in ti.static(range(9)):
            for j in ti.static(range(9)):
                res[i] += mat[i, j] * vec[j]
        return res

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.vel[i, j] = tm.vec2(0.0, 0.0)
            self.rho[i, j] = 1.0
            # 初始化直接使用 BGK equilibrium 填充 f_old, f_new 作為起點是沒問題的
            # 這裡保留舊邏輯即可，或者也可以用 MRT convert
            v = tm.vec2(0.0, 0.0)
            feq = ti.Vector([0.0] * 9, dt=ti.f32)
            # 簡單計算 f_eq 給初始化
            usq = 0.0
            for k in ti.static(range(9)):
                eu = v.x * self.e[k, 0] + v.y * self.e[k, 1]
                feq[k] = self.w[k] * 1.0 * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq)

            self.f_old[i, j] = feq
            self.f_new[i, j] = feq

    @ti.func
    def collide_and_stream(self, i: int, j: int):
        # [FIX] 使用 if-else 結構取代 return，解決 Taichi 編譯錯誤

        # 1. 如果是邊界 (最外圈)，直接跳過碰撞，保留舊值
        # 這些點的數值會在 step() 之後的 apply_bc() 中被正確覆蓋
        if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
            self.f_new[i, j] = self.f_old[i, j]

        else:
            # 2. 如果是內部節點 (非邊界)，執行正常的 LBM 運算

            # --- (A) 串流 (Streaming) ---
            f_local = ti.Vector([0.0] * 9, dt=ti.f32)
            for k in ti.static(range(9)):
                # 因為已經排除了最外圈，這裡直接讀取鄰居不會越界
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                f_local[k] = self.f_old[ip, jp][k]

            # --- (B) 障礙物處理 (Bounce-back) ---
            if self.mask[i, j] > 0.5:
                self.f_new[i, j] = ti.Vector(
                    [
                        f_local[0],
                        f_local[3],
                        f_local[4],
                        f_local[1],
                        f_local[2],
                        f_local[7],
                        f_local[8],
                        f_local[5],
                        f_local[6],
                    ]
                )
            else:
                # --- (C) MRT 碰撞核心 ---
                rho = 0.0
                vel = tm.vec2(0.0, 0.0)
                for k in ti.static(range(9)):
                    rho += f_local[k]
                    vel += tm.vec2(self.e[k, 0], self.e[k, 1]) * f_local[k]

                self.rho[i, j] = rho
                # 避免除以零的安全保護
                if rho > 1e-6:
                    vel /= rho
                self.vel[i, j] = vel

                # MRT 轉換與鬆弛
                m = self.mat_vec_mul(self.M, f_local)
                m_eq = self.get_moments_equilibrium(rho, vel)

                m_post = ti.Vector([0.0] * 9, dt=ti.f32)
                for k in ti.static(range(9)):
                    m_post[k] = m[k] - self.S[k] * (m[k] - m_eq[k])

                self.f_new[i, j] = self.mat_vec_mul(self.M_inv, m_post)

    @ti.kernel
    def step(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            self.collide_and_stream(i, j)

        # 更新 f_old (這步不能省，除非做雙緩衝優化，但目前先求穩)
        for i, j in ti.ndrange(self.nx, self.ny):
            self.f_old[i, j] = self.f_new[i, j]

    @ti.kernel
    def apply_bc(self, current_step: int):
        # ---------------------------------------------------------
        # 1. 速度緩升 (Ramp-up) 機制
        # ---------------------------------------------------------
        ramp = 1.0
        if current_step < 500:
            ramp = float(current_step) / 500.0

        # [修正點 1] 直接計算入口速度向量 (Vector)，而不是純量
        # self.bc_value[0] 已經是 [0.05, 0.0]，直接乘上 ramp 即可
        inlet_v = self.bc_value[0] * ramp

        # ---------------------------------------------------------
        # 2. 左側入口 (Inlet) - MRT 一致性
        # ---------------------------------------------------------
        for j in ti.ndrange(self.ny):
            if self.mask[0, j] == 0:
                # [修正點 2] 直接賦值向量，不需要再用 tm.vec2 包裝
                self.vel[0, j] = inlet_v
                self.rho[0, j] = 1.0

                # [修正點 3] 傳入正確的向量給平衡態函數
                m_eq = self.get_moments_equilibrium(1.0, inlet_v)
                f_boundary = self.mat_vec_mul(self.M_inv, m_eq)

                self.f_old[0, j] = f_boundary
                self.f_new[0, j] = f_boundary

        # ---------------------------------------------------------
        # 3. 右側出口 (Outlet) - 零梯度外推 (維持不變)
        # ---------------------------------------------------------
        for j in ti.ndrange(self.ny):
            if self.mask[self.nx - 1, j] == 0:
                self.rho[self.nx - 1, j] = self.rho[self.nx - 2, j]
                self.vel[self.nx - 1, j] = self.vel[self.nx - 2, j]

                self.f_old[self.nx - 1, j] = self.f_old[self.nx - 2, j]
                self.f_new[self.nx - 1, j] = self.f_old[self.nx - 1, j]

        # ---------------------------------------------------------
        # 4. 頂底邊界 (Top/Bottom) - 自由滑移 (維持不變)
        # ---------------------------------------------------------
        for i in ti.ndrange(self.nx):
            # --- 底部邊界 ---
            if self.mask[i, 0] == 0:
                self.vel[i, 0].y = 0.0

                rho = self.rho[i, 0]
                vel = self.vel[i, 0]

                m_eq = self.get_moments_equilibrium(rho, vel)
                f_boundary = self.mat_vec_mul(self.M_inv, m_eq)

                self.f_old[i, 0] = f_boundary
                self.f_new[i, 0] = f_boundary

            # --- 頂部邊界 ---
            if self.mask[i, self.ny - 1] == 0:
                self.vel[i, self.ny - 1].y = 0.0

                rho = self.rho[i, self.ny - 1]
                vel = self.vel[i, self.ny - 1]

                m_eq = self.get_moments_equilibrium(rho, vel)
                f_boundary = self.mat_vec_mul(self.M_inv, m_eq)

                self.f_old[i, self.ny - 1] = f_boundary
                self.f_new[i, self.ny - 1] = f_boundary

    # ... get_stats 維持不變 ...
    def get_stats(self) -> Dict[str, Any]:
        vel_np = self.vel.to_numpy()
        rho_np = self.rho.to_numpy()
        vel_mag = np.linalg.norm(vel_np, axis=-1)
        max_v = np.max(vel_mag)

        status = "healthy"
        if np.isnan(max_v) or np.max(rho_np) > 1.5:  # 放寬一點判斷
            status = "diverged"
        elif max_v < 0.00001:
            status = "blocked"

        return {
            "status": status,
            "max_v": float(max_v),
            "avg_v": float(np.mean(vel_mag[self.mask.to_numpy() == 0])),
            "re_max": float(max_v * self.nx / self.niu_lbm),  # 簡單估算
            "ma_max": float(max_v / 0.577),
        }
