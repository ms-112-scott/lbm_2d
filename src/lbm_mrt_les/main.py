import taichi as ti
import numpy as np
import os
import sys
import matplotlib.pyplot as plt  # 新增繪圖庫
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
from lbm2d_mrt_les import LBM2D_MRT_LES
from visualization import LBMVisualizer
from VideoRecorder import VideoRecorder

ti.init(arch=ti.gpu)


def main(config_path):
    # ------------------------------------------------
    # 1. 初始化 Solver
    # ------------------------------------------------
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config = utils.load_config(config_path)
    mask = utils.create_mask(config)

    solver = LBM2D_MRT_LES(config, mask_data=mask)
    solver.init()
    solver.Re = solver.check_re()

    # ------------------------------------------------
    # [策略監督] 計算 Flow-through Time (FTT)
    # ------------------------------------------------
    inlet_vel_vec = config["boundary_condition"]["value"][0]
    u_inlet = np.linalg.norm(inlet_vel_vec)

    if u_inlet < 1e-6:
        print("[Warning] Inlet velocity is nearly zero. Defaulting to 100,000 steps.")
        max_simulation_steps = 100000
    else:
        # 計算由左至右穿過一次所需的步數
        steps_per_pass = int(solver.nx / u_inlet)
        target_passes = 10  # DFG 建議跑久一點以觀察週期性 (建議至少 10-20 passes)
        max_simulation_steps = steps_per_pass * target_passes

        print("=" * 40)
        print(f"   SIMULATION STRATEGY: {target_passes} PASSES")
        print("=" * 40)
        print(f"Domain Length (nx) : {solver.nx}")
        print(f"Inlet Velocity (U) : {u_inlet:.4f}")
        print(f"Steps per Pass     : {steps_per_pass}")
        print(f"Target Total Steps : {max_simulation_steps}")
        print("=" * 40)

    # ------------------------------------------------
    # 2. 初始化 Visualizer & GUI & Output
    # ------------------------------------------------
    max_size = config.get("display", {}).get("max_size", 1024)
    viz = LBMVisualizer(
        nx=solver.nx,
        ny=solver.ny,
        viz_sigma=config["simulation"].get("visualization_gaussian_sigma", 1.0),
        max_display_size=max_size,
    )
    display_res = viz.get_display_size()
    gui = ti.GUI("Room Jet Flow (DFG Validation)", res=display_res)

    # 準備輸出路徑
    if "foler_paths" in config and "output" in config["foler_paths"]:
        out_dir = config["foler_paths"]["output"]
    else:
        out_dir = config.get("output", {}).get("video_dir", "./output")
    os.makedirs(out_dir, exist_ok=True)

    # 錄影設定
    recorder = None
    try:
        video_path = os.path.join(
            out_dir, f"Re{int(solver.Re)}_nx{solver.nx}_ForceTest.mp4"
        )
        recorder = VideoRecorder(
            video_path, width=display_res[0], height=display_res[1], fps=30
        )
        recorder.start()
        print(f"[Video] Saving to: {video_path}")
    except Exception as e:
        print(f"[Warning] VideoRecorder init failed: {e}")
        recorder = None

    # --- [新增] 力學數據容器 ---
    history_fx = []
    history_fy = []
    history_steps = []

    print("--- Simulation Started ---")

    # ------------------------------------------------
    # 3. 主迴圈
    # ------------------------------------------------
    current_steps = 0

    with tqdm(total=max_simulation_steps, unit="step", desc="LBM Progress") as pbar:

        while gui.running and current_steps < max_simulation_steps:

            # A. 物理步進
            solver.run_step(solver.steps_per_frame)
            current_steps += solver.steps_per_frame

            # --- [新增] 獲取並記錄力 ---
            # 注意：get_force 返回的是 numpy array [fx, fy]
            forces = solver.get_force()
            fx, fy = forces[0], forces[1]

            history_fx.append(fx)
            history_fy.append(fy)
            history_steps.append(current_steps)

            # 更新進度條文字顯示當前受力 (即時監控)
            pbar.set_postfix(Fx=f"{fx:.4e}", Fy=f"{fy:.4e}")
            pbar.update(solver.steps_per_frame)

            # B. 獲取數據 & 渲染
            vel, mask_data = solver.get_physical_fields()
            img_gui = viz.process_frame(vel, mask_data)

            gui.set_image(img_gui)
            gui.show()

            if recorder:
                img_video = np.transpose(img_gui, (1, 0, 2))
                recorder.write_frame(img_video)

    # ------------------------------------------------
    # 4. 結束處理與繪圖
    # ------------------------------------------------
    if recorder:
        recorder.stop()

    print(f"\n--- Simulation Completed. Generating Force Plots... ---")

    # 轉換為 Numpy 陣列以便繪圖
    history_steps = np.array(history_steps)
    history_fx = np.array(history_fx)
    history_fy = np.array(history_fy)

    # --- [新增] 儲存原始數據 (方便後續做 FFT 分析) ---
    data_path = os.path.join(out_dir, f"force_data_Re{int(solver.Re)}.npz")
    np.savez(data_path, steps=history_steps, fx=history_fx, fy=history_fy)
    print(f"[Data] Raw force data saved to: {data_path}")

    # --- [新增] 繪製力學曲線圖 ---
    plt.figure(figsize=(12, 8))

    # Subplot 1: Drag Force (Fx)
    plt.subplot(2, 1, 1)
    plt.plot(history_steps, history_fx, label="Drag Force (Fx)", color="blue")
    plt.title(f"Hydrodynamic Forces on Cylinder (Re={int(solver.Re)})")
    plt.ylabel("Force (Lattice Units)")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()

    # Subplot 2: Lift Force (Fy)
    plt.subplot(2, 1, 2)
    plt.plot(history_steps, history_fy, label="Lift Force (Fy)", color="red")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Force (Lattice Units)")
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.legend()

    # 儲存圖片
    plot_path = os.path.join(out_dir, f"force_plot_Re{int(solver.Re)}.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()  # 關閉 plot 釋放記憶體

    print(f"[Plot] Force history plot saved to: {plot_path}")
    print(f"[Info] Check the plot. If Re=100, Fy should oscillate sinusoidally!")

    gui.close()


if __name__ == "__main__":
    configs = ["config_DFGBenchmark2D-1.yaml"]  # 預設讀取 DFG 驗證設定
    path_prefix = "src/lbm_mrt_les/configs"  # 請確認您的路徑是否正確

    # 如果有參數傳入，優先使用參數
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        # 否則跑預設列表
        for config_name in configs:
            full_path = f"{path_prefix}/{config_name}"
            # 防呆檢查
            if os.path.exists(full_path):
                main(full_path)
            elif os.path.exists(config_name):  # 檢查是否在當前目錄
                main(config_name)
            else:
                print(f"Error: Config file not found: {full_path}")
