"""Interactive horse ASCII art animation demo."""

import os
import time

from reme.horse import _mirror_frame


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def running_horse():
    """Run a galloping horse animation across the terminal."""
    # 定义两帧动画，模拟腿部动作
    frame_1 = r"""
      >>\.
     /_  )`.
    /  _)`^)`.   _.---.
   (_,' \  `^----      `.
         |              |
         \              /
         / \  /___   / \
        /  /  |   \  \  |
    """

    frame_2 = r"""
      >>\.
     /_  )`.
    /  _)`^)`.   _.---.
   (_,' \  `^----      `.
         |              |
         \              /
        //  /   ___  /  |
       / / /   |   \ \  |
    """

    frame_1 = _mirror_frame(frame_1)
    frame_2 = _mirror_frame(frame_2)

    frames = [frame_1, frame_2]
    distance = 0

    try:
        while True:
            # 1. 清理屏幕
            clear_screen()

            # 2. 获取当前帧（通过取余数在两帧之间切换）
            current_frame = frames[distance % 2]

            # 3. 增加左侧空格，产生向右移动的效果
            indent = " " * distance

            # 4. 打印带缩进的每一行
            print("\n" * 5)  # 顶部留白
            for line in current_frame.split("\n"):
                # 只有非空行才打印，避免格式错乱
                if line.strip() != "":
                    print(indent + line)
                else:
                    print()

            # 5. 打印地面
            print("-" * (distance + 40))

            # 6. 更新距离并暂停
            distance += 1
            time.sleep(0.2)  # 控制速度，0.2秒一帧

            # 跑到屏幕边缘重置（可选）
            if distance > 60:
                distance = 0

    except KeyboardInterrupt:
        print("\n马儿休息了。(程序已停止)")


if __name__ == "__main__":
    running_horse()
