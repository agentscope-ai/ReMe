"""Horse Easter egg: fireworks, galloping horse animation, and a blessing."""

import math
import random
import shutil
import sys
import time


def _mirror_frame(frame: str) -> str:
    """Mirror ASCII art horizontally."""
    mirror_map = str.maketrans(r"()/\<>[]{}", r")(\/><][}{")
    lines = frame.split("\n")
    max_len = max(len(line) for line in lines) if lines else 0
    mirrored = []
    for line in lines:
        padded = line.ljust(max_len)
        reversed_line = padded[::-1].translate(mirror_map)
        mirrored.append(reversed_line)
    return "\n".join(mirrored)


def _play_horse_easter_egg() -> None:
    """Play the /horse Easter egg: fireworks, galloping horse, and a blessing."""
    cols = shutil.get_terminal_size((80, 24)).columns
    rows = shutil.get_terminal_size((80, 24)).lines

    # -- Fireworks animation (~4 seconds at 8 fps = 32 frames) --
    firework_colors = [
        "\033[91m",  # red
        "\033[93m",  # yellow
        "\033[92m",  # green
        "\033[96m",  # cyan
        "\033[95m",  # magenta
        "\033[94m",  # blue
    ]
    reset = "\033[0m"
    particles_chars = ["*", ".", "o", "+", "x", "'", "`"]

    class Firework:
        """A single firework burst with radial particles."""

        def __init__(self, cx: int, cy: int, color: str, birth: int):
            self.cx = cx
            self.cy = cy
            self.color = color
            self.birth = birth
            self.num = random.randint(12, 20)
            self.angles = [random.uniform(0, 2 * math.pi) for _ in range(self.num)]
            self.speeds = [random.uniform(0.5, 1.5) for _ in range(self.num)]
            self.chars = [random.choice(particles_chars) for _ in range(self.num)]

        def particles(self, frame: int):
            """Return list of (x, y, char) particle positions for the given frame."""
            age = frame - self.birth
            if age < 0 or age > 10:
                return []
            pts = []
            for i in range(self.num):
                r = self.speeds[i] * age
                px = self.cx + int(r * math.cos(self.angles[i]) * 2)  # *2 for aspect ratio
                py = self.cy + int(r * math.sin(self.angles[i]))
                if 0 <= px < cols and 0 <= py < rows - 1:
                    pts.append((px, py, self.chars[i]))
            return pts

    # Hide cursor
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()

    try:
        fireworks: list[Firework] = []
        total_frames = 32
        for f in range(total_frames):
            # Spawn new fireworks periodically
            if f % 4 == 0:
                cx = random.randint(10, cols - 10)
                cy = random.randint(2, rows // 2)
                color = random.choice(firework_colors)
                fireworks.append(Firework(cx, cy, color, f))

            # Build frame buffer (blank)
            buf: dict[tuple[int, int], tuple[str, str]] = {}
            for fw in fireworks:
                for px, py, ch in fw.particles(f):
                    buf[(px, py)] = (fw.color, ch)

            # Render
            sys.stdout.write("\033[H\033[2J")  # clear screen
            for y in range(rows - 1):
                line_parts: list[str] = []
                x = 0
                for x_pos in sorted(px for (px, py) in buf if py == y):
                    if x_pos >= x:
                        line_parts.append(" " * (x_pos - x))
                        color, ch = buf[(x_pos, y)]
                        line_parts.append(f"{color}{ch}{reset}")
                        x = x_pos + 1
                sys.stdout.write("".join(line_parts) + "\n")
            sys.stdout.flush()
            time.sleep(1 / 8)  # 8 fps

        # Prune old fireworks
        fireworks.clear()

        # -- Horse ASCII art (bold yellow) --
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

        mirrored_1 = _mirror_frame(frame_1)
        mirrored_2 = _mirror_frame(frame_2)

        bold_yellow = "\033[1;33m"
        sys.stdout.write("\033[H\033[2J")  # clear

        # Short galloping animation (8 cycles)
        for i in range(8):
            sys.stdout.write("\033[H\033[2J")
            horse = mirrored_1 if i % 2 == 0 else mirrored_2
            indent = " " * (i * 3)
            print("\n" * 3)
            for line in horse.split("\n"):
                if line.strip():
                    print(f"{bold_yellow}{indent}{line}{reset}")
            print(f"{bold_yellow}{'-' * min(i * 3 + 40, cols - 1)}{reset}")
            sys.stdout.flush()
            time.sleep(0.2)

        # -- Random blessing --
        blessings = [
            ("\u9a6c\u5230\u6210\u529f", "Succeed immediately"),
            ("\u9f99\u9a6c\u7cbe\u795e", "Full of vitality"),
            ("\u4e07\u9a6c\u5954\u817e", "Thousands of horses galloping"),
            ("\u9a6c\u4e0d\u505c\u8e44", "Never stop striving"),
            ("\u5feb\u9a6c\u52a0\u97ad", "Full speed ahead"),
            ("\u4e00\u9a6c\u5f53\u5148", "Take the lead"),
        ]
        cn, en = random.choice(blessings)
        print()
        print(f"{bold_yellow}  {cn} - {en}{reset}")
        print(f"{bold_yellow}  Happy Year of the Horse 2026!{reset}")
        print()

    finally:
        # Restore cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
