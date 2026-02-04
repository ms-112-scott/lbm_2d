import taichi as ti
import numpy as np
import sys
import os
import time

# 匯入自定義模組
from lbm2d_mrt_les import lbm_solver_les
from utils import load_config, Applying_BC, Applying_Mask, color_map
from VideoRecorder import VideoRecorder

# 初始化 Taichi (使用 GPU)
ti.init(arch=ti.gpu)


def main():
    # ==========================================
    # 1. 讀取設定與初始化
    # ==========================================
    config_path = "src/lbm_bgk/config.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = load_config(config_path)
    sim_cfg = config["simulation"]

    # 取得原始網格尺寸
    nx = sim_cfg["nx"]
    ny = sim_cfg["ny"]
    steps_per_frame = sim_cfg.get("steps_per_frame", 50)

    print(f"--- Simulation Setup ---")
    print(f"Grid: {nx} x {ny}")
    print(f"Model: MRT-LBM + Smagorinsky LES")

    # ==========================================
    # 2. 視窗尺寸設定
    # ==========================================
    # [修正] 直接使用模擬解析度，避免縮放造成的錯誤與效能損耗
    disp_w, disp_h = nx, ny
    print(f"Window: {disp_w}x{disp_h} (Native Resolution)")

    # ==========================================
    # 3. 初始化 Solver 與工具
    # ==========================================
    solver = lbm_solver_les(config)

    # Mask
    mask_mgr = Applying_Mask(config)
    solver.load_mask(mask_mgr.mask)

    # BC
    bc_mgr = Applying_BC(config)

    # GUI
    gui = ti.GUI(sim_cfg["name"], res=(disp_w, disp_h))

    # 錄影設定
    video_dir = config["output"].get("video_dir", "./output")
    video_name = os.path.join(video_dir, f"{sim_cfg['name']}_{int(time.time())}.mp4")
    save_video = config["output"].get("save_video", False)

    recorder = VideoRecorder(video_name, width=nx, height=ny, fps=30)
    if save_video:
        recorder.start()

    # 視覺化資源
    img_field = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))
    cmap_data = color_map("coolwarm")
    colormap = ti.Vector.field(3, dtype=ti.f32, shape=len(cmap_data))
    for i in range(len(cmap_data)):
        colormap[i] = ti.Vector(cmap_data[i])

    # ==========================================
    # 4. 主迴圈
    # ==========================================
    vis_mode = "vorticity"
    print("\nControls: 'v': Vel, 'c': Curl, 's': Screenshot, 'q': Quit")

    # --- Warmup ---
    if sim_cfg["warmup_steps"] > 0:
        print(f"Warming up ({sim_cfg['warmup_steps']} steps)...")
        for _ in range(0, sim_cfg["warmup_steps"], steps_per_frame):
            bc_mgr.update_time(steps_per_frame)
            u_in, v_in = bc_mgr.get_inlet_velocity()
            solver.run_step(steps_per_frame, u_in, v_in)
        print("Warmup done.")

    while gui.running:
        bc_mgr.update_time(steps_per_frame)
        u_in, v_in = bc_mgr.get_inlet_velocity()

        solver.run_step(steps_per_frame, u_in, v_in)

        if vis_mode == "velocity":
            solver.get_velocity_img(img_field, colormap)
        else:
            solver.get_vorticity_img(img_field, colormap)

        # [核心修正]
        # 1. 移除 .resize()，因為它回傳 None。
        # 2. 因為上面我們將 GUI 解析度設為 (nx, ny)，這裡直接轉 Numpy 即可完美匹配。
        img_np = img_field.to_numpy()

        gui.set_image(img_np)

        # 繪製 Sponge Layer
        if solver.use_sponge and solver.sponge_width > 0:
            ratio = solver.sponge_width / nx
            gui.rect(
                topleft=[1.0 - ratio, 1.0],
                bottomright=[1.0, 0.0],
                color=0xFF0000,
            )
            gui.text("SPONGE", pos=[1.0 - ratio, 0.95], color=0xFFFFFF, font_size=20)

        gui.show()

        # 錄影處理
        if save_video:
            # 轉置處理: (Width, Height) -> (Height, Width) 供影片編碼器使用
            img_video = np.transpose(img_np, (1, 0, 2))[::-1, :, :]
            recorder.write_frame(img_video)

        # 鍵盤輸入
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == "q":
                gui.running = False
            elif gui.event.key == "v":
                vis_mode = "velocity"
            elif gui.event.key == "c":
                vis_mode = "vorticity"
            elif gui.event.key == "s":
                gui.show(f"screenshot_{int(time.time())}.png")

    if save_video:
        recorder.stop()
    print("Done.")


if __name__ == "__main__":
    main()
