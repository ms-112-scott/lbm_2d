import cv2
import numpy as np
import os
import sys


def calculate_temporal_average_from_video(
    video_path, output_img_path, show_progress=True
):
    """
    讀取影片檔案，計算所有影格的時間平均值，並儲存為一張圖片。

    Args:
        video_path (str): 輸入影片檔案路徑 (.mp4, .avi 等)
        output_img_path (str): 輸出平均圖片的路徑 (.png, .jpg)
        show_progress (bool): 是否在終端顯示進度條

    Returns:
        bool: 成功返回 True, 失敗返回 False
    """

    # 1. 檢查影片是否存在
    if not os.path.exists(video_path):
        print(f"[Error] Video file not found: {video_path}")
        return False

    # 2. 打開影片捕捉器
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return False

    # 獲取影片資訊
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"--- Video Analysis Start ---")
    print(f"Input: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Total Frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print("Calculating temporal average...")

    avg_accumulator = None
    frame_count = 0

    # 3. 逐幀處理迴圈
    while True:
        ret, frame = cap.read()

        # 如果讀不到影格 (影片結束)
        if not ret:
            break

        frame_count += 1

        # [關鍵]: 將影像轉為 float32 進行累加，避免 overflow，並確保精度
        # frame 預設是 BGR 通道，我們會對三個通道分別做平均
        frame_float = frame.astype(np.float32)

        if avg_accumulator is None:
            # 初始化累積器 (大小與第一幀相同)
            avg_accumulator = frame_float
        else:
            # 累加
            avg_accumulator += frame_float

        # 顯示簡易進度
        if show_progress and frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            sys.stdout.write(
                f"/rProcessing: {progress:.1f}% ({frame_count}/{total_frames})"
            )
            sys.stdout.flush()

    if show_progress:
        print()  # 換行

    # 4. 計算平均值
    if frame_count == 0:
        print("[Error] No frames read from the video.")
        cap.release()
        return False

    print(f"Finished processing {frame_count} frames.")
    # 除以總幀數
    avg_img_float = avg_accumulator / frame_count

    # 5. 轉換回圖像格式 (uint8, 0-255)
    # 使用 clip 確保數值不會超出範圍，雖然理論上平均值不會超出
    avg_img_uint8 = np.clip(avg_img_float, 0, 255).astype(np.uint8)

    # 確保輸出目錄存在
    output_dir = os.path.dirname(output_img_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 儲存結果
    # 建議存為 PNG 以保留無損品質
    cv2.imwrite(output_img_path, avg_img_uint8)
    print(f"--- [Success] Average image saved to: {output_img_path} ---")

    cap.release()
    return True


# ==========================================
# 使用範例
# ==========================================
if __name__ == "__main__":
    # 這裡替換成你實際的影片路徑
    # 假設你最新的影片是這個：
    input_video = "src/lbm_mrt_les/output/Re719_nx1600_5Passes copy.mp4"

    # 自動生成輸出檔名 (把 .mp4 換成 _AVG.png)
    output_image = input_video.rsplit(".", 1)[0] + "_AVG.png"

    # 呼叫函數
    calculate_temporal_average_from_video(input_video, output_image)
