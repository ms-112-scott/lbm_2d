import yaml
import os
import glob
import taichi as ti
from tqdm import tqdm
from .utils import init_csv_log, get_processed_masks
from .engine import run_single_case

# 全域初始化 GPU
ti.init(arch=ti.gpu, offline_cache=False)  # 設為 False 則不會在硬碟產生任何快取檔案


def main():
    """批次模擬主程式入口"""
    lbm_2d_path = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(lbm_2d_path, '..', '..'))

    # 載入設定
    with open(os.path.join(root_dir, "config.yaml"), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 初始化 Log 與過濾機制
    log_path = os.path.join(
        root_dir, "output", cfg["simulation"]["name"], "_batch_run_result.csv"
    )
    init_csv_log(log_path)
    done_set = get_processed_masks(log_path)

    # 任務掃描
    mask_files = sorted(
        glob.glob(os.path.join(root_dir, cfg["obstacle"]["mask_dir"], "*.png"))
    )
    todo_tasks = [
        p
        for p in mask_files
        if os.path.splitext(os.path.basename(p))[0] not in done_set
    ]

    print(
        f"[*] 任務狀態: 總數 {len(mask_files)} | 已完成 {len(done_set)} | 剩餘 {len(todo_tasks)}"
    )

    # 執行任務
    for path in tqdm(todo_tasks, desc="批次進度"):
        run_single_case(cfg, path, root_dir, log_path)


if __name__ == "__main__":
    main()
