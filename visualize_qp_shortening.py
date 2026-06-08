"""Visualize MERLIN QP shortening choices and QP magnetization directions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from model import (
    MERLIN_BODY_DCC_FIRST,
    MERLIN_BODY_DCC_LAST,
    MERLIN_DEFAULT_QP_SHORT_BLOCKS,
    MERLIN_MAGNET_COLORS,
    MERLIN_QUADRANTS,
    MERLIN_STD_LZ,
    merlin_magnet_table,
    normalize_qp_short_blocks,
)


DEFAULT_QP_RETRACTION = 8.0
DEFAULT_GAP = 15.0
SUPPORTED_SIDE_VIEW_ROWS = frozenset({"Q1", "Q4"})


def _normalize_qp_retraction(qp_retraction: float) -> float:
    try:
        qp_retraction = float(qp_retraction)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"qp_retraction must be numeric, got {qp_retraction!r}"
        ) from exc
    if qp_retraction < 0:
        raise ValueError(f"qp_retraction must be non-negative, got {qp_retraction}")
    if qp_retraction > MERLIN_STD_LZ:
        raise ValueError(
            f"qp_retraction must not exceed MERLIN_STD_LZ={MERLIN_STD_LZ}, "
            f"got {qp_retraction}"
        )
    return qp_retraction


def _arrow_symbol(m_unit: list[float]) -> str:
    if abs(m_unit[2]) > abs(m_unit[1]):
        return "↑" if m_unit[2] > 0 else "↓"
    return "→" if m_unit[1] > 0 else "←"


def _row_center_z(gap: float, z_sign: float) -> float:
    return z_sign * (gap / 2.0 + MERLIN_STD_LZ / 2.0)


def _block_bottom(row_center: float, height: float, z_sign: float) -> float:
    standard_bottom = row_center - MERLIN_STD_LZ / 2.0
    standard_top = row_center + MERLIN_STD_LZ / 2.0
    if z_sign > 0:
        return standard_top - height
    return standard_bottom


def visualize_qp_shortening(
    gap: float = DEFAULT_GAP,
    z: float = 0.0,
    mode: str = "parallel",
    qp_retraction: float = DEFAULT_QP_RETRACTION,
    qp_short_blocks=None,
    *,
    quadrants: Iterable[str] = ("Q1", "Q4"),
    figsize: Sequence[float] | None = None,
    p_alpha: float = 0.2,
):
    """Return a side-view Matplotlib figure showing QP shortening and magnetization.

    Parameters
    ----------
    gap
        Vertical gap between Q1 and Q4 inner faces, in mm.
    z
        MERLIN phase in mm. Phase-shifted rows use this in the same sense as
        ``build_merlin``.
    mode
        Included for ``build_merlin`` call-site compatibility. Only
        ``"parallel"`` is supported.
    qp_retraction
        QP shortening/retraction value in mm.
    qp_short_blocks
        DCC body-block indices to render as QP-shortened magnets. If omitted,
        the repo's default QP block set is used.
    quadrants
        MERLIN quadrant rows rendered in the side view. The default shows Q1
        and Q4, the two rows visible from the +x side.
    figsize
        Matplotlib figure size in inches. If omitted, a repo-default size is
        chosen from the body-block count.
    """
    if mode != "parallel":
        raise NotImplementedError("Only parallel mode is supported")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    qp_retraction = _normalize_qp_retraction(qp_retraction)

    quadrants = tuple(quadrants)
    for quadrant in quadrants:
        if quadrant not in MERLIN_QUADRANTS:
            raise ValueError(f"Unknown MERLIN quadrant: {quadrant}")
        if quadrant not in SUPPORTED_SIDE_VIEW_ROWS:
            raise ValueError("Only Q1 and Q4 are supported for this side view")

    qp_blocks = normalize_qp_short_blocks(qp_short_blocks)

    fig_width = max(10.0, 0.05 * (MERLIN_BODY_DCC_LAST - MERLIN_BODY_DCC_FIRST + 1))
    if figsize is None:
        figsize = (fig_width, 2.6)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=333)
    x_bounds = []
    z_bounds = []
    label_positions = {}

    for quadrant in quadrants:
        quadrant_cfg = MERLIN_QUADRANTS[quadrant]
        z_sign = quadrant_cfg["z_sign"]
        row_center = _row_center_z(gap, z_sign)
        phase = z if quadrant_cfg["phase_shifted"] else 0.0
        for entry in merlin_magnet_table(
            quadrant,
            phase=phase,
            qp_short_blocks=qp_blocks,
        ):
            block = entry["dcc_block"]
            center_y = entry["y_center_mm"]
            width_y = entry["ly_mm"]
            is_qp = entry["is_qp_short"]
            if is_qp:
                height = MERLIN_STD_LZ - qp_retraction
                zorder = 2
            else:
                height = MERLIN_STD_LZ
                zorder = 1
            bottom = _block_bottom(row_center, height, z_sign)
            left = center_y - width_y / 2.0
            alpha = 1.0 if is_qp else p_alpha

            ax.add_patch(
                Rectangle(
                    (left, bottom),
                    width_y,
                    height,
                    facecolor=MERLIN_MAGNET_COLORS[entry["type"]],
                    edgecolor="black",
                    alpha=alpha,
                    zorder=zorder,
                )
            )

            ax.text(
                center_y,
                row_center,
                _arrow_symbol(entry["m_unit"]),
                ha="center",
                va="center",
                alpha=alpha,
                zorder=4,
                fontname="DejaVu Sans Mono",
                fontweight="bold",
                fontsize=8,
            )
            x_bounds.extend((left, left + width_y))
            z_bounds.extend((bottom, bottom + height))
            if is_qp:
                label_positions.setdefault(block, center_y)

    for block, center_y in label_positions.items():
        ax.text(
            center_y,
            max(z_bounds) + 4.0,
            str(block),
            ha="center",
            va="bottom",
        )

    ax.set_xlim(min(x_bounds), max(x_bounds))
    ax.set_ylim(min(z_bounds) - 4.0, max(z_bounds) + 14.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.set_aspect("equal", adjustable="box")
    return fig
