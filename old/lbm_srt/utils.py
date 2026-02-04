import os
import csv
import numpy as np
import taichi as ti
from PIL import Image
from typing import Set


def load_mask_from_png(path, nx, ny):
    """
    讀取 PNG 遮罩並轉換為符合 Taichi 座標系的 NumPy 陣列。
    座標轉換邏輯：
    1. PIL 讀取 [height, width] (原點在左上)
    2. flipud 翻轉為原點在左下
    3. .T 轉置為 [nx, ny] 以符合 Taichi field shape
    """
    # 讀取並縮放影像
    img = Image.open(path).convert("L").resize((nx, ny), Image.NEAREST)
    mask_np = np.array(img)

    # 解決上下顛倒：影像 (0,0) 在左上 -> 物理 (0,0) 在左下
    # flipud: 上下翻轉; .T: 轉置 [y, x] -> [x, y]
    mask_correct = np.flipud(mask_np).T

    # 0 為牆壁 (Mask=1.0), 255 為空氣 (Mask=0.0)
    return np.where(mask_correct < 128, 1.0, 0.0).astype(np.float32)


def save_snapshot(output_dir, step, vel_field, gui_img=None, save_npy=True):
    """
    儲存模擬結果。
    """
    path_base = os.path.join(output_dir, f"step_{step:06d}")

    # 儲存原始速度場數據 (Numpy 格式)
    if save_npy:
        # 注意：存檔時 Taichi field 會自動轉回 Numpy
        np.save(f"{path_base}.npy", vel_field.to_numpy())

    # 儲存可視化影像 (PNG)
    if gui_img is not None:
        # ti.tools.imwrite 預期輸入為 (width, height, channels)
        ti.tools.imwrite(gui_img, f"{path_base}.png")


def load_mask_from_png(path: str, nx: int, ny: int) -> np.ndarray:
    """載入並修正 PNG 遮罩，使其符合 Taichi 座標系 (左下角為原點)"""
    img = Image.open(path).convert("L").resize((nx, ny), Image.NEAREST)
    # flipud 處理上下顛倒，.T 處理 [y,x] -> [x,y] 轉置
    return np.where(np.flipud(np.array(img)).T < 128, 1.0, 0.0).astype(np.float32)


def init_csv_log(log_path: str):
    """初始化批次任務的 CSV 紀錄表"""
    if not os.path.exists(log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Mask_Name",
                    "Steps",
                    "Avg_Vel",
                    "Max_Ma",
                    "Max_Re",
                    "Time(s)",
                    "Status",
                ]
            )


def get_processed_masks(log_path: str) -> Set[str]:
    """獲取已完成的案例名稱集合，避免重複執行"""
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add(row[0])
    return processed
