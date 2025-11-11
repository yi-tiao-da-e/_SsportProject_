import cv2
import tkinter as tk
from tkinter import filedialog

class VideoInput:
    def __init__(self):
        pass

    def select_video_file(self):
        """选择视频文件并返回路径"""
        return filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")]
        )
    
    def test_video_file(self, file_path):
        """测试视频文件是否可以打开"""
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret
    
    def get_video_capture(self, file_path):
        """获取视频捕获对象"""
        return cv2.VideoCapture(file_path)