# 基於 Taichi 語言的 Lattice Boltzmann Method (LBM) 流體解算器
# 作者 : Wang (hietwll@gmail.com)
# 註解/翻譯 : Gemini

import sys
import matplotlib
import numpy as np
from matplotlib import cm

import taichi as ti
import taichi.math as tm

# 初始化 Taichi，使用 GPU 進行平行運算加速
ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name,  # 模擬案例名稱
        nx,  # 計算域寬度 (x方向網格數)
        ny,  # 計算域高度 (y方向網格數)
        niu,  # 流體運動黏滯係數 (Kinematic Viscosity)
        bc_type,  # 邊界條件類型 [左, 上, 右, 下]: 0 -> Dirichlet (固定速度); 1 -> Neumann (零梯度/流出)
        bc_value,  # 邊界條件數值: 如果 bc_type = 0，這裡指定速度向量 [u, v]
        cy=0,  # 是否放置圓柱障礙物 (1: 是, 0: 否)
        cy_para=[0.0, 0.0, 0.0],  # 圓柱參數: [圓心x, 圓心y, 半徑]
    ):
        self.name = name
        self.nx = nx  # 依慣例，LBM 中 dx = dy = dt = 1.0 (格子單位)
        self.ny = ny
        self.niu = niu

        # 計算鬆弛時間 (Relaxation time) tau
        # 公式: niu = (cs^2) * (tau - 0.5)，其中聲速平方 cs^2 = 1/3
        # 移項得: tau = 3 * niu + 0.5
        self.tau = 3.0 * niu + 0.5
        self.inv_tau = 1.0 / self.tau

        # 定義 Taichi 場 (Fields)
        self.rho = ti.field(float, shape=(nx, ny))  # 密度場
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))  # 速度場 (二維向量)
        self.mask = ti.field(
            float, shape=(nx, ny)
        )  # 固體遮罩 (1.0 代表障礙物，0.0 代表流體)

        # 分佈函數 f (Distribution Functions)
        # f_old: 上一時刻的分佈, f_new: 更新後的分佈
        # 每個網格點有 9 個分量 (D2Q9)
        self.f_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.f_new = ti.Vector.field(9, float, shape=(nx, ny))

        # D2Q9 模型的權重 w
        # 中心點: 4/9, 軸向: 1/9, 對角線: 1/36
        self.w = (
            ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        )

        # D2Q9 模型的離散速度向量 e (c_i)
        # 0:中心, 1-4:軸向(右,上,左,下), 5-8:對角線(右上,左上,左下,右下)
        self.e = ti.types.matrix(9, 2, int)(
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]
        )

        # 處理邊界條件輸入
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.cy = cy
        self.cy_para = tm.vec3(cy_para)

    @ti.func  # Taichi 函數，計算平衡分佈函數 f_eq
    def f_eq(self, i, j):
        # 計算過程: f_eq = w_k * rho * [1 + 3(e.u) + 4.5(e.u)^2 - 1.5(u.u)]
        eu = self.e @ self.vel[i, j]  # e 點乘 u (向量投影)
        uv = tm.dot(self.vel[i, j], self.vel[i, j])  # u 點乘 u (速度大小平方)
        return self.w * self.rho[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel  # Taichi kernel，GPU平行運算入口
    def init(self):
        self.vel.fill(0)  # 初始速度設為 0
        self.rho.fill(1)  # 初始密度設為 1
        self.mask.fill(0)

        # 對所有網格進行初始化
        for i, j in self.rho:
            # 初始狀態設為平衡態
            self.f_old[i, j] = self.f_new[i, j] = self.f_eq(i, j)

            # 如果啟用了圓柱障礙物，設置 mask
            if self.cy == 1:
                # 判斷點是否在圓內: (x-x0)^2 + (y-y0)^2 <= r^2
                if (i - self.cy_para[0]) ** 2 + (
                    j - self.cy_para[1]
                ) ** 2 <= self.cy_para[2] ** 2:
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self):  # LBM 核心方程式：碰撞與串流
        # 平行遍歷內部網格 (避開邊界，邊界由 apply_bc 處理)
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):  # ti.static 在編譯時展開迴圈
                # --- 串流 (Streaming) 步驟的逆向思考 ---
                # 我們要計算位置 (i, j) 在方向 k 的新分佈 f_new。
                # 這個粒子是從反方向的鄰居流過來的。
                # 來源座標 (ip, jp) = (i, j) - e[k]
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]

                # 計算來源位置的平衡分佈
                feq = self.f_eq(ip, jp)

                # --- LBGK 碰撞算子 (Collision) ---
                # 公式: f_new = (1 - 1/tau) * f_old + (1/tau) * f_eq
                # 這裡直接結合了 "從鄰居讀取 (Stream)" 和 "進行碰撞 (Collide)"
                self.f_new[i, j][k] = (1 - self.inv_tau) * self.f_old[ip, jp][k] + feq[
                    k
                ] * self.inv_tau

    @ti.kernel
    def update_macro_var(self):  # 更新巨觀變量 (密度 rho, 速度 u, v)
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0
            self.vel[i, j] = 0, 0

            for k in ti.static(range(9)):
                # 更新 f_old 為下一步做準備
                self.f_old[i, j][k] = self.f_new[i, j][k]

                # 計算密度: rho = sum(f_k)
                self.rho[i, j] += self.f_new[i, j][k]

                # 計算動量: rho * u = sum(e_k * f_k)
                self.vel[i, j] += (
                    tm.vec2(self.e[k, 0], self.e[k, 1]) * self.f_new[i, j][k]
                )

            # 速度正規化: u = (rho * u) / rho
            self.vel[i, j] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):  # 施加邊界條件
        # 處理左右邊界
        for j in range(1, self.ny - 1):
            # 左邊界: 方向索引0 (Left), 邊界x=0, 鄰居x=1
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # 右邊界: 方向索引2 (Right), 邊界x=nx-1, 鄰居x=nx-2
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # 處理上下邊界
        for i in range(self.nx):
            # 上邊界: 方向索引1 (Top), 邊界y=ny-1, 鄰居y=ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # 下邊界: 方向索引3 (Bottom), 邊界y=0, 鄰居y=1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # 處理圓柱障礙物邊界 (內部固體邊界)
        # 注意: 在 CUDA 後端，將 if 判斷放在迴圈內通常比分離迴圈更快
        for i, j in ti.ndrange(self.nx, self.ny):
            if self.cy == 1 and self.mask[i, j] == 1:
                self.vel[i, j] = 0, 0  # 固體邊界速度為 0 (No-slip)

                # 尋找最近的流體鄰居節點，用於外推邊界條件
                inb = 0
                jnb = 0
                if i >= self.cy_para[0]:
                    inb = i + 1
                else:
                    inb = i - 1
                if j >= self.cy_para[1]:
                    jnb = j + 1
                else:
                    jnb = j - 1

                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        # 這裡使用的是非平衡態外推格式 (Non-Equilibrium Extrapolation Method)

        if outer == 1:  # 處理外部邊界 (計算域的四個邊)
            if self.bc_type[dr] == 0:
                # Dirichlet: 強制設定邊界速度 (例如入口速度)
                self.vel[ibc, jbc] = self.bc_value[dr]

            elif self.bc_type[dr] == 1:
                # Neumann: 速度梯度為零 (例如出口)，速度等於鄰居速度
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        # 密度設為鄰居的密度 (壓力邊界近似)
        self.rho[ibc, jbc] = self.rho[inb, jnb]

        # 核心公式: f_boundary = f_eq(boundary) + (f_old(neighbor) - f_eq(neighbor))
        # 意義: 假設非平衡部分 (f - f_eq) 在邊界和鄰居之間是不變的
        self.f_old[ibc, jbc] = (
            self.f_eq(ibc, jbc) - self.f_eq(inb, jnb) + self.f_old[inb, jnb]
        )

    def solve(self):
        # 初始化 GUI 視窗，高度設為 2 倍是因為要同時顯示兩個圖
        gui = ti.GUI(self.name, (self.nx, 2 * self.ny))
        self.init()

        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # 每個渲染幀執行 10 次模擬步，加快視覺流動速度
            for _ in range(10):
                self.collide_and_stream()  # 碰撞與串流
                self.update_macro_var()  # 更新巨觀量
                self.apply_bc()  # 應用邊界條件

            ## --- 視覺化部分 ---
            # 從 GPU 獲取速度場數據到 CPU (numpy)
            vel = self.vel.to_numpy()

            # 計算渦度 (Vorticity) = dv/dx - du/dy
            # 這裡使用 numpy 的 gradient 函數計算梯度
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]

            # 計算速度大小 (Velocity Magnitude)
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5

            ## 設定顏色映射 (Color Map)
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

            # 將渦度轉換為圖像 (上方圖)
            vor_img = cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap
            ).to_rgba(vor)
            # 將速度大小轉換為圖像 (下方圖, 使用 plasma 配色)
            vel_img = cm.plasma(vel_mag / 0.15)

            # 拼接兩張圖並顯示
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()


if __name__ == "__main__":
    # 從命令列參數讀取案例編號，預設為 0
    flow_case = 0 if len(sys.argv) < 2 else int(sys.argv[1])

    if flow_case == 0:  # 卡門渦街 (Von Karman Vortex Street)
        # 雷諾數 Re = U*D/niu = 200
        lbm = lbm_solver(
            "Karman Vortex Street",
            801,  # nx
            201,  # ny
            0.01,  # niu (黏滯係數)
            [0, 0, 1, 0],  # 邊界: 左(入流), 上(固壁), 右(出流), 下(固壁)
            [
                [0.05, 0.0],  # 左(入流)
                [0.0, 0.0],  # 上(固壁)
                [0.0, 0.0],  # 右(出流)
                [0.0, 0.0],  # 下(固壁)
            ],  # 左邊界速度 U=0.1
            1,  # 啟用圓柱
            [160.0, 100.0, 20.0],
        )  # 圓柱位置與半徑
        lbm.solve()

    elif flow_case == 1:  # 頂蓋驅動穴流 (Lid-driven Cavity Flow)
        # 雷諾數 Re = U*L/niu = 1000
        lbm = lbm_solver(
            "Lid-driven Cavity Flow",
            256,
            256,
            0.0255,  # 黏滯係數調整以匹配 Re=1000
            [0, 0, 0, 0],  # 四面都是 Dirichlet 邊界
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )  # 只有頂部(index 1)有速度 U=0.1
        lbm.solve()

    else:
        print(
            "Invalid flow case ! Please choose from 0 (Karman Vortex Street) and 1 (Lid-driven Cavity Flow)."
        )
