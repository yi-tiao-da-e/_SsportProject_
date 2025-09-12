print("启动运动分析应用...")

from video_io.input_handler import InputHandler

def main():
    print("启动运动分析应用...")
    input_handler = InputHandler()
    input_handler.start()  # 启动输入处理（用户交互+视频读取）

if __name__ == "__main__":

    main()