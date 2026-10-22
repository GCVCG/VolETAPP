"""Boards detected per threshold phi (image step of find_scale_headless.py only)."""
import contextlib, io, os, sys
from find_scale_headless import Generate_Real_Scale

D = "/media/HDD_4TB_1/amughrabi/data/MTF_Challenge"
for item in sys.argv[1:]:
    g = Generate_Real_Scale.__new__(Generate_Real_Scale)
    g.chess_w, g.chess_h = 4, 3
    g.image_root = f"{D}/{item}/dense/images"
    names = sorted(os.listdir(g.image_root)[:50])
    paths = [f"{g.image_root}/{n}" for n in names]
    counts = {}
    for T in (190, 200, 210, 220, 230, 240, 250):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                counts[T] = len(g.chessboard_detection(paths, names, T)[2])
            except (SystemExit, IndexError):
                counts[T] = "<10"
    print(item, len(names), counts, flush=True)
