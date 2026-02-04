import taichi as ti
import taichi.math as tm
import numpy as np
from typing import Dict, Any


@ti.data_oriented
class LBMSolver:
    """
    Lattice Boltzmann Method 求解器，使用 D2Q9 模型。
    """

    def __init__(self, cfg: Dict[str, Any], mask_data: np.ndarray):
        """
        [Python 邏輯] 初始化求解器，分配硬體資源。
        注意：這裡不要加 @ti.kernel
        """
        # 基礎參數設定
        self.nx: int = cfg["simulation"]["nx"]
        self.ny: int = cfg["simulation"]["ny"]
        self.niu_lbm: float = cfg["simulation"]["niu"]
        self.inv_tau: float.py = 1.0 / (3.0 * self.niu_lbm + 0.5)

        inlet_vel: float = cfg["boundaries"]["values"][0][0]
        self.Tc: int = int(self.nx / max(inlet_vel, 0.001))

        # Taichi 場域定義 (分配記憶體)
        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.mask = ti.field(float, shape=(self.nx, self.ny))
        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))

        # LBM 常數
        self.w = ti.types.vector(9, float)(
            4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36
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

        # 邊界條件值 (從 Config 讀取)
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(
            np.array(cfg["boundaries"]["values"], dtype=np.float32)
        )

        # 載入遮罩
        self.mask.from_numpy(mask_data)

    @ti.kernel
    def init(self):
        """
        [GPU 內核] 初始化場域數值。
        """
        for i, j in self.rho:
            self.vel[i, j] = tm.vec2(0.0, 0.0)
            self.rho[i, j] = 1.0
            feq = self.get_f_eq(i, j)
            self.f_old[i, j] = feq
            self.f_new[i, j] = feq

    @ti.func
    def get_f_eq(self, i: int, j: int):
        """[GPU 函數] 計算平衡態分佈函數"""
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def apply_bc(self):
        """
        [GPU 內核] 套用邊界條件 (入口、出口、側壁)。
        """
        # 這裡根據你的 config 邏輯實作邊界
        # 範例：左側入口 (假設入口在 x=0)
        for j in range(self.ny):
            self.vel[0, j] = self.bc_value[0]  # 使用入口速度
            self.rho[0, j] = 1.0
            self.f_old[0, j] = self.get_f_eq(0, j)

    @ti.kernel
    def step(self):
        """[GPU 內核] 執行 Collision & Streaming"""
        # 1. Collision & Streaming
        for i, j in ti.ndrange(self.nx, self.ny):
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                if 0 <= ip < self.nx and 0 <= jp < self.ny:
                    feq_p = self.get_f_eq(ip, jp)
                    self.f_new[i, j][k] = self.f_old[ip, jp][k] + self.inv_tau * (
                        feq_p[k] - self.f_old[ip, jp][k]
                    )

        # 2. 更新物理量與處理 Bounce-back
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.mask[i, j] == 1.0:
                self.vel[i, j] = tm.vec2(0.0, 0.0)
                self.rho[i, j] = 1.0
                self.f_old[i, j] = self.get_f_eq(i, j)
            else:
                rho_val = 0.0
                vel_val = tm.vec2(0.0, 0.0)
                for k in ti.static(range(9)):
                    f_val = self.f_new[i, j][k]
                    self.f_old[i, j][k] = f_val
                    rho_val += f_val
                    vel_val += tm.vec2(self.e[k, 0], self.e[k, 1]) * f_val

                self.rho[i, j] = ti.max(0.6, ti.min(rho_val, 1.4))
                self.vel[i, j] = vel_val / self.rho[i, j]

    def get_stats(self) -> Dict[str, Any]:
        """[Python 邏輯] 獲取統計資訊"""
        vel_np = self.vel.to_numpy()
        rho_np = self.rho.to_numpy()
        vel_mag = np.linalg.norm(vel_np, axis=-1)
        max_v = np.max(vel_mag)

        status = "healthy"
        if np.isnan(max_v) or np.max(rho_np) > 1.3:
            status = "diverged"
        elif max_v < 0.0001:
            status = "blocked"

        return {
            "status": status,
            "max_v": float(max_v),
            "avg_v": float(np.mean(vel_mag[self.mask.to_numpy() == 0])),
            "re_max": float(max_v / self.niu_lbm),
            "ma_max": float(max_v / 0.577),
        }
