import numpy as np
import cv2
import os
import random


def generate_geometry_masks(
    num_images=10,
    width=800,
    height=200,
    output_dir="output/masks",
    mode="mix",  # 可選: "circle", "rect", "mix"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    margin = 50
    min_size = 15
    x_range = (margin, width - margin)
    y_range = (margin, height - margin)

    for i in range(num_images):
        img = np.ones((height, width), dtype=np.uint8) * 255
        cx = random.randint(x_range[0], x_range[1])
        cy = random.randint(y_range[0], y_range[1])

        # 計算安全半徑
        dist_to_boundary = min(
            cx - margin, (width - margin) - cx, cy - margin, (height - margin) - cy
        )
        size = random.randint(min_size, int(max(min_size, dist_to_boundary)))

        # 根據 mode 決定形狀邏輯
        current_shape = mode
        if mode == "mix":
            current_shape = random.choice(["circle", "rect"])

        if current_shape == "circle":
            # --- 繪製圓形 ---
            cv2.circle(img, (cx, cy), size, 0, -1)
            shape_name = "Circle"

        elif current_shape == "rect":
            # --- 繪製正方形 (隨機旋轉 0~180度) ---
            angle = random.uniform(0, 180)
            shape_name = f"Rect_{angle:.1f}"

            # 定義頂點與旋轉
            pts = np.array([[-size, -size], [size, -size], [size, size], [-size, size]])
            theta = np.radians(angle)
            rotation_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )

            # 旋轉與平移
            rotated_pts = (rotation_matrix @ pts.T).T + np.array([cx, cy])
            cv2.fillPoly(img, [rotated_pts.astype(np.int32)], 0)

        # 儲存
        file_path = os.path.join(output_dir, f"{shape_name}_{i:03d}.png")
        cv2.imwrite(file_path, img)
        print(f"[Generated] {file_path}")


if __name__ == "__main__":
    # 使用範例：
    generate_geometry_masks(num_images=5, mode="circle")  # 全圓形
    generate_geometry_masks(num_images=5, mode="rect")  # 全正方形(含旋轉)
    # generate_geometry_masks(num_images=10, mode="mix")  # 隨機混合
