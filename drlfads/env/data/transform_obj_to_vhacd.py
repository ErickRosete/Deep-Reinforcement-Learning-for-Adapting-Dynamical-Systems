import pybullet as p
from pathlib import Path

p.connect(p.DIRECT)
dir = Path(__file__).parents[0]
name_in = str(dir / "peg_in_hole_board/board_hard.obj")
name_out = str(dir / "peg_in_hole_board/board_hard_vhacd.obj")
name_log = str(dir / "log.txt")
p.vhacd(name_in, name_out, name_log, alpha=0.04, resolution=64000000)