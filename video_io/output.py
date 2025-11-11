import pickle
import os
import tkinter as tk
from tkinter import filedialog

class VideoOutput:
    def __init__(self):
        pass

    def save_landmarks_data(self, landmarks_3d):
        """保存姿态数据到文件"""
        if not landmarks_3d:
            return None
        
        file_path = filedialog.asksaveasfilename(
            title="保存姿态数据",
            defaultextension=".pkl",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")]
        )
        
        if file_path:
            with open(file_path, 'wb') as f:
                pickle.dump(landmarks_3d, f)
            return file_path
        return None