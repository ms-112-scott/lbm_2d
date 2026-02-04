import matplotlib
import numpy as np
from matplotlib import cm
import taichi as ti
import taichi.math as tm
import sys
import os
from VideoRecorder import VideoRecorder
from scipy.ndimage import gaussian_filter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

ti.init(arch=ti.gpu)


@ti.data_oriented
class lbm_solver_les:
    def __init__(self, config, mask_data=None):
        self.config = config
        sim_cfg = config["simulation"]
        bc_cfg = config["boundary_condition"]

        self.name = sim_cfg["name"]
        self.nx = sim_cfg["nx"]
        self.ny = sim_cfg["ny"]
        self.niu = sim_cfg["niu"]
        self.steps_per_frame = sim_cfg.get("steps_per_frame", 10)
        self.frame_count = ti.field(int, shape=())
        self.warmup_steps = sim_cfg["warmup_steps"]

        # --- 更新：讀取 Config 中的 Ghost Moments S，若無則預設 1.0 ---
        self.S_other = sim_cfg["ghost_moments_s"]
        # --- 更新：讀取 Config 中的高斯模糊 Sigma，若無則預設 1.0 ---
        self.viz_sigma = sim_cfg["visualization_gaussian_sigma"]

        self.C_smag = sim_cfg["smagorinsky_constant"]
        self.Cs_sq_factor = 18.0 * (self.C_smag**2)

        self.bc_type_list = bc_cfg["type"]
        self.bc_value_list = bc_cfg["value"]
        self.tau_0 = 3.0 * self.niu + 0.5

        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.mask = ti.field(float, shape=(self.nx, self.ny))

        if mask_data is not None:
            self.mask.from_numpy(mask_data.astype(np.float32))
        else:
            self.mask.fill(0.0)

        self.f_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.f_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))

        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(self.bc_type_list, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(self.bc_value_list, dtype=np.float32))

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

        M_inv_np = np.linalg.inv(M_np)
        self.M_field = ti.field(float, shape=(9, 9))
        self.M_inv_field = ti.field(float, shape=(9, 9))
        self.M_field.from_numpy(M_np)
        self.M_inv_field.from_numpy(M_inv_np)

        # 使用讀取到的 S_other
        self.S_base = ti.types.vector(9, float)(
            0.0,
            self.S_other,
            self.S_other,
            0.0,
            self.S_other,
            0.0,
            self.S_other,
            0.0,
            0.0,
        )

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
            f_temp = ti.types.vector(9, float)(0.0)
            for k in ti.static(range(9)):
                ip, jp = i - self.e[k, 0], j - self.e[k, 1]
                f_temp[k] = self.f_old[ip, jp][k]

            m = ti.types.vector(9, float)(0.0)
            for r in ti.static(range(9)):
                val = 0.0
                for c in ti.static(range(9)):
                    val += self.M_field[r, c] * f_temp[c]
                m[r] = val

            rho_l = m[0]
            u_l, v_l = 0.0, 0.0
            if rho_l > 0:
                u_l, v_l = m[3] / rho_l, m[5] / rho_l

            m_eq = self.get_meq(rho_l, u_l, v_l)
            neq_7 = m[7] - m_eq[7]
            neq_8 = m[8] - m_eq[8]
            momentum_neq_mag = tm.sqrt(neq_7 * neq_7 + neq_8 * neq_8)

            tau_eff = self.tau_0
            if self.C_smag > 0.001:
                term_inside = (
                    self.tau_0**2 + (self.Cs_sq_factor * momentum_neq_mag) / rho_l
                )
                tau_eddy = 0.5 * (tm.sqrt(term_inside) - self.tau_0)
                tau_eff = self.tau_0 + tau_eddy

            s_eff = 1.0 / tau_eff
            S_local = self.S_base
            S_local[7] = s_eff
            S_local[8] = s_eff

            m_star = m - S_local * (m - m_eq)

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
        u_vec = self.bc_value_list[0]
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

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, self.ny * 2))
        self.init()
        self.Re = self.check_re()

        recorder = VideoRecorder(
            f"output_Re{int(self.Re)}_nx{self.nx}.mp4", self.nx, self.ny * 2, fps=30
        )
        recorder.start()

        colors = [
            (1, 1, 0),
            (0.953, 0.490, 0.016),
            (0, 0, 0),
            (0.176, 0.976, 0.529),
            (0, 1, 1),
        ]
        my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
        my_cmap.set_bad(color="grey")

        try:
            while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
                for _ in range(self.steps_per_frame):
                    self.collide_and_stream()
                    self.update_macro_var()
                    self.apply_bc()

                vel_raw = self.vel.to_numpy()
                vel_x = gaussian_filter(vel_raw[:, :, 0], sigma=self.viz_sigma)
                vel_y = gaussian_filter(vel_raw[:, :, 1], sigma=self.viz_sigma)

                ugrad = np.gradient(vel_x)
                vgrad = np.gradient(vel_y)
                vor = ugrad[1] - vgrad[0]
                vel_mag = np.sqrt(vel_x**2 + vel_y**2)

                mask_np = self.mask.to_numpy()
                vor[mask_np > 0] = np.nan

                vor_img = cm.ScalarMappable(
                    norm=matplotlib.colors.Normalize(vmin=-0.03, vmax=0.03),
                    cmap=my_cmap,
                ).to_rgba(vor)[:, :, :3]

                vel_img = cm.plasma(vel_mag / 0.15)[:, :, :3]

                for d in range(3):
                    vor_img[:, :, d] = np.where(mask_np == 1, 0.5, vor_img[:, :, d])
                    vel_img[:, :, d] = np.where(mask_np == 1, 0.5, vel_img[:, :, d])

                # 原始拼接: (nx, ny*2, 3) -> 適合 GUI
                img_gui = np.concatenate((vel_img, vor_img), axis=1)
                gui.set_image(img_gui)
                gui.show()

                # --- 錄影轉置 ---
                # GUI 是 (Width, Height)，Video 需要 (Height, Width)
                img_video = np.transpose(img_gui, (1, 0, 2))
                recorder.write_frame(img_video)

        except Exception as e:
            print(f"Simulation crashed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            recorder.stop()


if __name__ == "__main__":
    config_path = "src/lbm_bgk/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = utils.load_config(config_path)
    nx, ny = config["simulation"]["nx"], config["simulation"]["ny"]
    mask_cfg = config.get("mask", {})
    mask = None

    if mask_cfg.get("enable"):
        m_type = mask_cfg["type"]
        print(f"Generating Mask: {m_type}")

        # --- 3. 更新：Mask 生成邏輯分支 ---
        if m_type == "cylinder":
            p = mask_cfg["params"]
            mask = utils.create_cylinder_mask(nx, ny, p["cx"], p["cy"], p["r"])
        elif m_type == "room":  # 新增 room 判斷
            mask = utils.create_two_rooms_mask(nx, ny)

    solver = lbm_solver_les(config, mask_data=mask)
    solver.solve()
