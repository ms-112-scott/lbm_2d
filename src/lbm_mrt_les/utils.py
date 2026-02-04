import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt


def load_config(path="config.yaml"):
    """讀取 YAML 設定檔"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading config: {e}")
        sys.exit(1)


def print_reynolds_info(u_char, l_char, nu, shape_name="Characteristic Length"):
    """
    計算並列印雷諾數資訊
    Re = (U * L) / nu
    """
    re = (u_char * l_char) / nu

    print("=" * 40)
    print(f"   REYNOLDS NUMBER CALCULATION")
    print("=" * 40)
    print(f"Characteristic Velocity (U) : {u_char:.6f} (Lattice Speed)")
    print(f"Characteristic Length   (L) : {l_char:.2f}   ({shape_name})")
    print(f"Kinematic Viscosity     (nu): {nu:.6f}")
    print("-" * 40)
    print(f"LBM Reynolds Number (Re)    : {re:.2f}")
    print(f"Physical Reynolds Number    : {re:.2f} (Dimensionless)")
    print("=" * 40)

    return re


def plot_mask(mask):
    """視覺化 Mask"""
    plt.figure(figsize=(10, 5))
    plt.imshow(mask.T, cmap="gray_r", origin="lower")  # .T 是為了轉置讓 x 軸橫向
    plt.title("Two Rooms Layout (White=Wall, Black=Air)")
    plt.colorbar()
    plt.show()


def _create_cylinder_mask(nx, ny, cx, cy, r):
    """產生圓柱障礙物遮罩 (Mask)"""
    # 建立網格座標矩陣
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    # 計算每個點到圓心的距離平方
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    # 產生 Mask (圓內為 1, 圓外為 0)
    mask = np.where(dist_sq <= r**2, 1.0, 0.0)
    return mask


def _create_two_rooms_mask(nx, ny, shift_left=200):
    mask = np.zeros((nx, ny))

    # --- 參數設定 ---
    w = 12  # 牆壁厚度
    d_half = 18  # 開口寬度的一半

    # 原始邊距
    marginLR = 600
    marginTD = 80

    # --- [核心修改：往左移動] ---
    # 減少左邊距，增加右邊距，圖形就會整體向左靠
    x_start = marginLR - shift_left
    x_end = (nx - marginLR) - shift_left

    y_start = marginTD
    y_end = ny - marginTD

    # --- 中心點/中牆位置 ---
    # 這裡也要跟著 x_start 和 x_end 的範圍來計算，圖形才不會變形
    # 讓中牆維持在房間範圍內的 1/3 處
    room_width = x_end - x_start
    x_mid = x_start + int(room_width / 3)

    # Y 軸中心維持不變
    y_mid = ny // 2

    for i in range(nx):
        for j in range(ny):

            # 1. 定義牆壁位置
            is_left_wall = x_start <= i < x_start + w
            is_right_wall = x_end - w <= i < x_end

            # 中牆位置
            is_mid_wall = x_mid - w // 2 <= i < x_mid + w // 2

            is_top_wall = y_end - w <= j < y_end
            is_bottom_wall = y_start <= j < y_start + w

            # 2. 定義開口位置
            is_opening_zone = y_mid - d_half <= j < y_mid + d_half

            # 3. 繪製邏輯
            if (is_top_wall or is_bottom_wall) and (x_start <= i < x_end):
                mask[i, j] = 1.0

            if (
                (is_left_wall or is_right_wall or is_mid_wall)
                and (not is_opening_zone)
                and (y_start <= j < y_end)
            ):
                mask[i, j] = 1.0

    return mask


def create_mask(config):
    mask_cfg = config.get("mask", {})
    mask = None
    nx = config["simulation"]["nx"]
    ny = config["simulation"]["ny"]
    if mask_cfg.get("enable"):
        m_type = mask_cfg["type"]
        print(f"Generating Mask: {m_type}")

        # --- 3. 更新：Mask 生成邏輯分支 ---
        if m_type == "cylinder":
            p = mask_cfg["params"]
            mask = _create_cylinder_mask(nx, ny, p["cx"], p["cy"], p["r"])
        elif m_type == "room":  # 新增 room 判斷
            mask = _create_two_rooms_mask(nx, ny)

    return mask
