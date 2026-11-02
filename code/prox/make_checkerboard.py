"""렌즈 보정용 체커보드 A4 PDF 만들기 (Mac에서 실행, 인쇄용)
칸 25mm x 10x7칸 → 안쪽 코너 9x6 (calib_intrinsics.py 기본값 --board 9x6 --square 0.025)
인쇄 후 100mm 자와 칸 한 변을 실제로 재서, 다르면 --square에 잰 값을 넣는다.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

COLS, ROWS, SQ = 10, 7, 25.0          # 칸 수, 칸 크기(mm)
PW, PH = 297.0, 210.0                 # A4 가로(mm)
plt.rcParams["font.family"] = "AppleGothic"

fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PW); ax.set_ylim(PH, 0); ax.axis("off")
x0, y0 = (PW - COLS * SQ) / 2, 22.0
for r in range(ROWS):
    for c in range(COLS):
        if (r + c) % 2 == 0:
            ax.add_patch(Rectangle((x0 + c * SQ, y0 + r * SQ), SQ, SQ, color="black", lw=0))
ax.text(x0, 12, f"렌즈 보정용 체커보드 — 칸 {SQ:.0f}mm, {COLS}x{ROWS}칸 (안쪽 코너 {COLS - 1}x{ROWS - 1})",
        fontsize=10, va="center")
ax.text(x0, 17.5, "인쇄: '실제 크기(100%)'로, 맞춤·축소 끄기. 딱딱한 판에 평평하게 붙여 쓰기",
        fontsize=8, va="center", color="#444")
yb = y0 + ROWS * SQ + 8
ax.plot([x0, x0 + 100], [yb, yb], color="black", lw=1.2)
for i in range(11):
    h = 3 if i % 5 == 0 else 1.6
    ax.plot([x0 + i * 10] * 2, [yb, yb - h], color="black", lw=0.8)
ax.text(x0 + 104, yb, "← 100mm 자 (인쇄 후 재서 확인)", fontsize=8, va="center")
fig.savefig("checkerboard_A4_9x6_25mm.pdf")
print("saved checkerboard_A4_9x6_25mm.pdf")
