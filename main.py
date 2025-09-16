# main.py（最顶部！）
print("===== 检测main.py是否被执行 =====")  # 绝对输出，不依赖任何模块

import sys
from pathlib import Path

# 修复可能的路径问题（关键！）
sys.path.append(str(Path(__file__).parent))  # 将项目根目录加入Python路径

try:
    from video_io.input_handler import InputHandler
    print("✅ InputHandler导入成功")
except ImportError as e:
    print(f"❌ InputHandler导入失败：{str(e)}")
    sys.exit(1)  # 导入失败直接退出

def main():
    print("启动运动分析应用...")
    input_handler = InputHandler()
    input_handler.start()

if __name__ == "__main__":
    print("===== 进入main函数 =====")
    main()