import tkinter as tk
from tkinter import messagebox
import traceback
import time  # 用于获取系统时间（替代Tkinter）
from video_io.input_handler import InputHandler  # 根据项目结构调整导入路径


def main():
    """程序主入口，负责初始化环境、处理用户输入、启动分析流程"""
    print("=" * 50)
    print("=== 运动分析程序启动 ===")
    print("=" * 50)

    root = None
    input_handler = None

    try:
        # 1. 打印当前系统时间（使用time模块，避免创建Tk实例）
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"=== 当前系统时间：{current_time} ===")

        # 2. 初始化Tk主窗口（隐藏，避免显示空白窗口）
        root = tk.Tk()
        root.title("运动分析系统")
        root.withdraw()  # 隐藏主窗口，仅保留事件循环
        print("=== Tk主窗口初始化完成（已隐藏） ===")

        # 3. 创建输入处理器实例（传递主窗口作为父容器）
        input_handler = InputHandler(master=root)
        print("=== InputHandler实例创建成功 ===")

        # 4. 显示输入设置对话框（模态，阻塞直到用户操作完成）
        print("=== 显示输入设置对话框 ===")
        if input_handler.show_input_dialog():
            # 5. 用户选择了有效输入源，启动分析流程
            print(f"=== 用户选择输入源：{input_handler.input_source}（类型：{input_handler.input_type}） ===")
            print(f"=== 选中关节：{input_handler.selected_joints} ===")
            print("=== 启动运动分析流程 ===")
            input_handler.start()
        else:
            # 用户取消操作
            print("=== 用户取消了操作，程序退出 ===")
            return

    except ImportError as e:
        print("=" * 50)
        print(f"=== 导入错误：{e} ===")
        print("=== 请检查模块路径是否正确，或依赖是否安装 ===")
        traceback.print_exc()
    except RuntimeError as e:
        print("=" * 50)
        print(f"=== 运行时错误：{e} ===")
        print("=== 可能原因：Tk主窗口重复创建或显示环境问题 ===")
        traceback.print_exc()
    except Exception as e:
        print("=" * 50)
        print(f"=== 程序运行错误：{e} ===")
        print("=== 详细错误信息如下 ===")
        traceback.print_exc()
    finally:
        # 6. 启动Tk事件循环（处理窗口事件，必须放在最后）
        if root:
            print("=" * 50)
            print("=== 启动Tk事件循环 ===")
            root.mainloop()
        print("=" * 50)
        print("=== 程序结束 ===")


if __name__ == "__main__":
    main()