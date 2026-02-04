import os
import time
import csv
from PIL import Image
import numpy as np
import taichi as ti
from matplotlib import cm
from solver import LBMSolver
from utils import load_mask_from_png, save_snapshot
from tqdm import tqdm
from typing import Dict, Any


def run_single_case(
    cfg: Dict[str, Any], mask_path: str, base_dir: str, log_path: str
) -> str:
    """執行單個流體模擬案例"""
    start_t = time.time()
    mask_name = os.path.splitext(os.path.basename(mask_path))[0]

    # 初始化尺寸與資料
    nx, ny = Image.open(mask_path).size
    cfg["simulation"].update({"nx": nx, "ny": ny})
    lbm = LBMSolver(cfg, load_mask_from_png(mask_path, nx, ny))
    lbm.init()

    # 目錄準備
    case_dir = os.path.join(base_dir, "output", cfg["simulation"]["name"], mask_name)
    os.makedirs(case_dir, exist_ok=True)

    # 模擬參數
    step, final_status = 0, "Success"
    stop_step = cfg["simulation"]["stop_Tc_count"] * lbm.Tc
    gui = (
        ti.GUI(f"LBM - {mask_name}", (nx, 2 * ny))
        if not cfg["simulation"]["silent_mode"]
        else None
    )

    with tqdm(total=stop_step, desc=f" > {mask_name}", leave=False) as pbar:
        while step < stop_step:
            for _ in range(cfg["simulation"]["steps_per_batch"]):
                lbm.step()
                lbm.apply_bc()
            step += cfg["simulation"]["steps_per_batch"]
            pbar.update(cfg["simulation"]["steps_per_batch"])

            stats = lbm.get_stats()
            if stats["status"] != "healthy":
                final_status = f"Failed ({stats['status']})"
                break

            # 渲染與存檔
            if gui or cfg["simulation"].get("save_png"):
                img = _render_frame(lbm, cfg["boundaries"]["values"][0][0])
                if gui:
                    gui.set_image(img)
                    gui.show()
                if step % cfg["simulation"]["save_step"] == 0:
                    save_snapshot(
                        case_dir, step, lbm.vel, img, cfg["simulation"]["save_npy"]
                    )

    if gui:
        gui.close()
    _record_log(log_path, mask_name, step, stats, time.time() - start_t, final_status)
    return final_status


def _render_frame(lbm: LBMSolver, inlet_v: float) -> np.ndarray:
    """生成速度場與渦度場的對比圖"""
    v_np = lbm.vel.to_numpy()
    v_mag = np.linalg.norm(v_np, axis=-1)
    # 簡單渦度計算
    vor = np.zeros_like(v_mag)
    vor[1:-1, 1:-1] = (v_np[2:, 1:-1, 1] - v_np[:-2, 1:-1, 1]) - (
        v_np[1:-1, 2:, 0] - v_np[1:-1, :-2, 0]
    )

    img_v = cm.plasma(np.clip(v_mag / (inlet_v * 1.5), 0, 1))[:, :, :3]
    img_w = cm.RdBu_r(np.clip(vor * 20 + 0.5, 0, 1))[:, :, :3]
    return np.concatenate((img_v, img_w), axis=1)


def _record_log(path: str, name: str, step: int, stats: dict, dt: float, status: str):
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                name,
                step,
                f"{stats['avg_v']:.5f}",
                f"{stats['ma_max']:.3f}",
                f"{stats['re_max']:.1f}",
                f"{dt:.2f}",
                status,
            ]
        )
