import numpy as np
import cv2
import os
import random
from typing import Tuple


def generate_city_masks(
    num_images: int = 10,
    width: int = 800,
    height: int = 400,
    output_dir: str = "output/city_masks",
    style: str = "skyline",
    pad: int = 50,  # 擴張的像素寬度
):
    """
    生成城市遮罩：先計算核心內容 -> 擴大 Padding -> Resize 到目標尺寸。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 定義核心內容區域 (Core Content Size)
    # 我們假設內容佔據了目標尺寸減去兩倍 pad 的空間
    core_w = max(width - 2 * pad, 100)
    core_h = max(height - 2 * pad, 100)

    for i in range(num_images):
        # 建立核心畫布 (白色背景)
        img_core = np.ones((core_h, core_w), dtype=np.uint8) * 255

        if style == "skyline":
            # --- 模式 A: 城市天際線 (側面圖) ---
            num_buildings = random.randint(10, 20)
            for _ in range(num_buildings):
                w = random.randint(20, core_w // 5)
                h = random.randint(30, int(core_h * 0.8))
                x = random.randint(0, core_w - w)
                # 底部對齊核心區域底部
                cv2.rectangle(img_core, (x, core_h - h), (x + w, core_h), 0, -1)

                # 塔尖
                if random.random() > 0.7:
                    tw = random.randint(5, 15)
                    th = random.randint(10, 20)
                    cv2.rectangle(
                        img_core,
                        (x + w // 2 - tw // 2, core_h - h - th),
                        (x + w // 2 + tw // 2, core_h - h),
                        0,
                        -1,
                    )

        else:
            # --- 模式 B: 城市街區 (俯瞰圖) ---
            num_blocks = random.randint(30, 50)
            for _ in range(num_blocks):
                w, h = random.randint(15, 40), random.randint(15, 40)
                x, y = random.randint(w, core_w - w), random.randint(h, core_h - h)
                angle = random.choice([0, 90, 15, -15])

                # 計算旋轉矩形頂點
                rect = ((x, y), (w, h), angle)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.fillPoly(img_core, [box], 0)

        # 2. 擴大 Padding (使用 cv2.copyMakeBorder)
        # 在四周補上白色 (255) 的邊界
        img_padded = cv2.copyMakeBorder(
            img_core, pad, pad, pad, pad, borderType=cv2.BORDER_CONSTANT, value=255
        )

        # 3. Resize 回到目標尺寸
        # 使用 INTER_NEAREST 確保遮罩依然是純黑白，不會產生灰色過渡帶
        final_img = cv2.resize(
            img_padded, (width, height), interpolation=cv2.INTER_NEAREST
        )

        # 儲存
        file_path = os.path.join(output_dir, f"city_{style}_p{pad}_{i:03d}.png")
        cv2.imwrite(file_path, final_img)
        print(f"[Done] Saved: {file_path} | Size: {width}x{height}")


if __name__ == "__main__":
    # 測試生成：天際線風格
    # generate_city_masks(num_images=5, width=800, height=400, style="skyline", pad=60)
    # 測試生成：俯瞰圖風格
    generate_city_masks(num_images=5, width=800, height=400, style="topdown", pad=150)
