"""
Hinh so sanh breakdown thoi gian prefill: TokenSelect goc vs SCR.
Stacked horizontal bar, bang mau lavender (ordinal ramp da qua validator cua skill dataviz).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# --- so lieu do duoc (giay, cong don toan bo run; 616 macro-chunk x layer moi cau hinh) ---
# giay/sample, n=30, da quy doi ve luot khong bat profiler (he so 0.8242)
STAGES = [
    ("Chunk-plan",       0.006, 0.219),
    ("KV-cache store",   0.342, 0.057),
    ("Token retrieval",  4.497, 0.887),
    ("Wrapper setup",    0.399, 0.060),
    ("Attention compute",2.982, 2.796),
    ("Loop bookkeeping", 0.582, 0.115),
]
CONFIGS = ["TokenSelect (base)", "TokenSelect + SCR"]

# lavender ordinal ramp — validateOrdinal: monotone L, dL>=0.083, light-end 2.32:1, CVD dE 8.5
# ordinal ramp lavender, validateOrdinal: dL>=0.091, light-end 2.18:1, CVD dE 8.6 (PASS)
RAMP = ["#b6a5d9", "#9c86c5", "#8267b1", "#6a489c", "#512b82", "#381361"]

SURFACE = "#fcfcfb"
INK      = "#0b0b0b"
INK_2    = "#52514e"
INK_MUTE = "#807e79"

BAR_H = 0.34
GAP   = 0.022          # khe ~2px giua cac doan


def rounded(ax, x, y, w, h, color, left=False, right=False, r=0.055):
    """Chu nhat, chi bo tron dau ngoai cua thanh (theo mark spec)."""
    r = min(r, w / 2, h / 2)
    rl, rr = (r if left else 0), (r if right else 0)
    v = [(x + rl, y), (x + w - rr, y)]
    c = [Path.MOVETO, Path.LINETO]
    if rr:
        v += [(x + w, y), (x + w, y + rr)]; c += [Path.CURVE3, Path.CURVE3]
    v += [(x + w, y + h - rr)]; c += [Path.LINETO]
    if rr:
        v += [(x + w, y + h), (x + w - rr, y + h)]; c += [Path.CURVE3, Path.CURVE3]
    v += [(x + rl, y + h)]; c += [Path.LINETO]
    if rl:
        v += [(x, y + h), (x, y + h - rl)]; c += [Path.CURVE3, Path.CURVE3]
    v += [(x, y + rl)]; c += [Path.LINETO]
    if rl:
        v += [(x, y), (x + rl, y)]; c += [Path.CURVE3, Path.CURVE3]
    v += [(x + rl, y)]; c += [Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(v, c), facecolor=color, edgecolor="none", zorder=3))


fig, ax = plt.subplots(figsize=(5.5, 2.60))
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.30)

totals = [sum(s[1] for s in STAGES), sum(s[2] for s in STAGES)]
ypos = [1.00, 0.12]

for ci, ypos_i in enumerate(ypos):
    x = 0.0
    vals = [s[1 + ci] for s in STAGES]
    last = max(i for i, v in enumerate(vals) if v > 0)
    first = min(i for i, v in enumerate(vals) if v > 0)
    for si, v in enumerate(vals):
        if v <= 0:
            continue
        w = max(v - GAP, 0.012)
        rounded(ax, x, ypos_i, w, BAR_H, RAMP[si],
                left=(si == first), right=(si == last))
        # nhan truc tiep, chi cho doan du rong (selective direct labels)
        if v >= 0.75:
            ax.text(x + w / 2, ypos_i + BAR_H / 2, f"{v:.2f}s",
                    ha="center", va="center", fontsize=8.6, color="#ffffff",
                    fontweight="medium", zorder=4)
        x += v
    # tong o cuoi thanh
    ax.text(totals[ci] + 0.18, ypos_i + BAR_H / 2, f"{totals[ci]:.2f}s",
            ha="left", va="center", fontsize=9.6, color=INK, fontweight="bold", zorder=4)
    # ten cau hinh dat NGAY TREN thanh -> bo duoc dai trong ben trai
    ax.text(0.0, ypos_i + BAR_H + 0.075, CONFIGS[ci], ha="left", va="bottom",
            fontsize=9.0, color=INK)

# mui ten chenh lech tong, dat giua khe hai thanh
Y_ARROW = 0.85
ax.annotate("", xy=(totals[1], Y_ARROW), xytext=(totals[0], Y_ARROW),
            arrowprops=dict(arrowstyle="-|>", color=INK_MUTE, lw=1.2,
                            shrinkA=0, shrinkB=0))
ax.text((totals[0] + totals[1]) / 2, Y_ARROW - 0.055,
        f"-{totals[0]-totals[1]:.2f}s  ({(totals[1]/totals[0]-1)*100:.0f}%)",
        ha="center", va="top", fontsize=8.6, color=INK_2)

ax.set_xlim(-0.10, 9.85)
ax.set_ylim(-0.06, 1.60)
ax.set_yticks([])
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#dcdad4")
ax.tick_params(axis="x", colors=INK_2, labelsize=8.4, length=0, pad=4)
ax.set_xticks([0, 2, 4, 6, 8])
ax.set_xlabel("Prefill attention-path time per sample (s)", fontsize=8.8, color=INK_2, labelpad=5)
ax.grid(axis="x", color="#eceae4", lw=0.9, zorder=0)
ax.set_axisbelow(True)

# chu thich
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=RAMP[i], edgecolor="none")
           for i in range(len(STAGES))]
labels = [s[0] for s in STAGES]
# matplotlib xep legend theo COT; dao lai de doc trai->phai dung thu tu pipeline
NCOL, NROW = 3, 2
order = [r * NCOL + c for c in range(NCOL) for r in range(NROW)]
order = [i for i in order if i < len(handles)]
handles = [handles[i] for i in order]; labels = [labels[i] for i in order]
ax.legend(handles, labels, loc="upper center",
          bbox_to_anchor=(0.5, -0.26), ncol=NCOL, frameon=False,
          fontsize=7.4, labelcolor=INK_2, handlelength=0.8,
          handleheight=0.85, columnspacing=1.4, handletextpad=0.4)
for ext in ("png", "pdf"):
    fig.savefig(f"prefill_breakdown.{ext}", dpi=220, bbox_inches="tight",
                facecolor=SURFACE)
print("da luu prefill_breakdown.png / .pdf")
