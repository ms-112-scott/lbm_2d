import imageio_ffmpeg
import subprocess
import numpy as np


class VideoRecorder:
    def __init__(self, filename, width, height, fps=30):
        self.filename = filename
        # 確保寬高為偶數
        self.rec_width = width - 1 if width % 2 != 0 else width
        self.rec_height = height - 1 if height % 2 != 0 else height
        self.fps = fps
        self.is_recording = False
        self.process = None
        self.ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    def start(self):
        command = [
            self.ffmpeg_exe,
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.rec_width}x{self.rec_height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            self.filename,
        ]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
            self.is_recording = True
            print(f"--- [Video] Recording started: {self.filename} ---")
        except FileNotFoundError:
            print("--- [Error] FFMPEG not found. ---")

    def write_frame(self, img_array):
        if not self.is_recording or self.process is None:
            return
        # 裁切並轉為 uint8
        img_cropped = img_array[: self.rec_height, : self.rec_width, :]
        frame = (np.clip(img_cropped, 0, 1) * 255).astype(np.uint8)
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            self.stop()

    def stop(self):
        if self.is_recording and self.process:
            try:
                self.process.stdin.close()
                self.process.wait()
            except:
                pass
            self.is_recording = False
            self.process = None
            print(f"--- [Video] Saved. ---")
