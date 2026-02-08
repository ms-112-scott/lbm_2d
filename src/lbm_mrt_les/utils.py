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


def _create_rect_mask(nx, ny, cx, cy, r):
    """
    產生矩形障礙物遮罩 (Mask)
    x0, y0: 矩形左上角 (或起始點) 座標
    w, h: 矩形的寬度與高度
    """
    y, x = np.meshgrid(np.arange(ny), np.arange(nx))
    # 判斷點是否在矩形範圍內： x0 <= x < x0+w 且 y0 <= y < y0+h
    mask = np.where(
        (x >= cx - r / 2) & (x < cx + r / 2) & (y >= cy - r / 2) & (y < cy + r / 2),
        1.0,
        0.0,
    )
    return mask


def _create_two_rooms_mask(nx, ny, shift_left=50, angle_deg=20):
    mask = np.zeros((nx, ny))

    # --- 1. 原始參數設定 (保持不變) ---
    w = 6  # 牆壁厚度
    d_half = 8  # 開口寬度的一半
    marginLR = 350
    marginTD = 100

    # 原始邊距計算
    x_start = marginLR - shift_left
    x_end = (nx - marginLR) - shift_left
    y_start = marginTD
    y_end = ny - marginTD

    room_width = x_end - x_start
    x_mid = x_start + int(room_width / 3)
    y_mid = ny // 2

    # --- 2. 旋轉參數設定 (新增) ---
    # 將角度轉為弧度
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # 設定旋轉中心 (Pivot Point)
    # 建議設在「房間的中心」，這樣旋轉時房間才不會跑出畫面
    cx = (x_start + x_end) / 2
    cy = (y_start + y_end) / 2

    # --- 3. 迴圈檢查 (加入座標旋轉) ---
    for i in range(nx):
        for j in range(ny):

            # [核心修改]: 座標逆向旋轉
            # 我們要檢查畫布上的點 (i, j)，在旋轉前的原始空間是對應到哪裡
            dx = i - cx
            dy = j - cy

            # 旋轉公式 (Inverse Rotation):
            # 如果要把物體逆時針轉 20 度，等於座標軸順時針轉 20 度
            # x' = dx * cos + dy * sin
            # y' = -dx * sin + dy * cos
            local_x = dx * cos_t + dy * sin_t + cx
            local_y = -dx * sin_t + dy * cos_t + cy

            # --- 接下來的邏輯完全不變，只是把 i, j 換成 local_x, local_y ---

            # 1. 定義牆壁位置 (使用 local 座標)
            is_left_wall = x_start <= local_x < x_start + w
            is_right_wall = x_end - w <= local_x < x_end

            # 中牆位置
            is_mid_wall = x_mid - w // 2 <= local_x < x_mid + w // 2

            is_top_wall = y_end - w <= local_y < y_end
            is_bottom_wall = y_start <= local_y < y_start + w

            # 2. 定義開口位置
            is_opening_zone = y_mid - d_half <= local_y < y_mid + d_half

            # 用來限制左右牆與中牆的高度範圍
            in_y_range = y_start <= local_y < y_end
            # 用來限制上下牆的寬度範圍
            in_x_range = x_start <= local_x < x_end

            # 3. 繪製邏輯
            # 上下牆
            if (is_top_wall or is_bottom_wall) and in_x_range:
                mask[i, j] = 1.0

            # 垂直牆 (左、右、中) + 避開開口
            if (
                (is_left_wall or is_right_wall or is_mid_wall)
                and (not is_opening_zone)
                and in_y_range
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

        # --- 更新：Mask 生成邏輯分支 ---
        if m_type == "cylinder":
            p = mask_cfg["params"]
            mask = _create_cylinder_mask(nx, ny, p["cx"], p["cy"], p["r"])

        elif m_type == "rect":  # 新增 rect 判斷
            p = mask_cfg["params"]
            mask = _create_rect_mask(nx, ny, p["cx"], p["cy"], p["r"])

        elif m_type == "room":
            mask = _create_two_rooms_mask(nx, ny)

    return mask
