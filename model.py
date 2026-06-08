"""Models the QP-EPU90 at MERLIN."""

import operator

import radia as rad

# Undulator geometry parameters (all lengths are in [mm]).
MERLIN_PERIOD = 90.0  # Magnetic period.
MERLIN_NPER = 20  # Number of magnetic periods in the measured device.
MERLIN_GAPX = (
    1.0  # Provisional horizontal row separation from the generic APPLE-II example.
)

# Magnet block dimensions (all lengths are in [mm]).
MERLIN_STD_LX = 25.0  # Standard block horizontal size.
MERLIN_STD_LY = 22.35  # Standard block longitudinal size.
MERLIN_STD_LZ = 50.0  # Standard block vertical size.

MERLIN_CX = 3.0  # Horizontal size of the lower notch/step used by MagnetBlock.
MERLIN_CZ = 5.0  # Vertical size of the lower notch/step used by MagnetBlock.
MERLIN_TOP_CX = 5.5  # Horizontal size of the upper notch/step used by MagnetBlock.
MERLIN_TOP_CZ = 3.0  # Vertical size of the upper notch/step used by MagnetBlock.
MERLIN_TOP_SLOT_CAP_LZ = 8.0  # Full-height material above regular/end upper slots.
MERLIN_AIR = (
    MERLIN_PERIOD / 4.0 - MERLIN_STD_LY
)  # Longitudinal air space implied by period/4 minus block length.
MERLIN_END_EDGE_OUTER_LY = 6.30  # Magnetic width of the outer narrow tip magnet.
MERLIN_END_EDGE_INNER_LY = 10.95  # Longitudinal size of the inner narrow tip magnet.

# Long E span containing the internal gap plus the inner end magnet.
MERLIN_END_LONG_SPAN_GAP = 13.0
MERLIN_END_INNER_LY = 17.45  # Magnetic length of the inner end magnet.
MERLIN_END_LONG_SPAN_LY = MERLIN_END_LONG_SPAN_GAP + MERLIN_END_INNER_LY

# Material constants and Radia subdivision parameters.
# These are just taken from the generic APPLE-II example and have not been optimized.
MERLIN_BR = 1.27  # Remanent magnetization magnitude [T].
MERLIN_MU = [
    0.05,
    0.15,
]  # Linear material susceptibility parameters used by rad.MatLin.
MERLIN_NDIV = [3, 2, 3]  # Magnet subdivision counts along x, y, z.

# DCC drawing-arrow names mapped to local unit magnetization vectors.
MERLIN_ARROW_TO_LOCAL_UNIT = {  # Local DCC row coordinates are converted to global Radia axes per quadrant.
    "up": [0.0, 0.0, 1.0],  # Local +vertical arrow.
    "right": [0.0, 1.0, 0.0],  # Local +longitudinal arrow.
    "down": [0.0, 0.0, -1.0],  # Local -vertical arrow.
    "left": [0.0, -1.0, 0.0],  # Local -longitudinal arrow.
}

# DCC magnet-type arrow table for body magnets.
MERLIN_MAGNET_TYPE_TO_ARROW = {  # Magnetization directions in the local row coordinates.
    "A1": "down",  # Standard A1 body magnet points in -z.
    "A2": "up",  # Standard A2 body magnet points in +z.
    "B1": "left",  # Standard B1 body magnet points in -y.
    "B2": "right",  # Standard B2 body magnet points in +y.
    "QA1": "down",  # Quasi A1 body magnet follows A1 magnetization.
    "QA2": "up",  # Quasi A2 body magnet follows A2 magnetization.
    "QB1": "left",  # Quasi B1 body magnet follows B1 magnetization.
    "QB2": "right",  # Quasi B2 body magnet follows B2 magnetization.
}

# Display colors for standard, quasi-periodic, and E-end block labels.
MERLIN_MAGNET_COLORS = {  # Colors only affect the Radia viewer, not the field calculation.
    "A1": [0.20, 0.25, 0.95],  # Standard A1 block color.
    "A2": [0.20, 0.65, 0.95],  # Standard A2 block color.
    "B1": [0.25, 0.85, 0.25],  # Standard B1 block color.
    "B2": [0.85, 0.85, 0.25],  # Standard B2 block color.
    "QA1": [0.95, 0.25, 0.20],  # QP replacement for an A1-family vertical block.
    "QA2": [0.95, 0.45, 0.20],  # QP replacement for an A2-family vertical block.
    "QB1": [0.10, 0.95, 0.50],  # QP replacement for a B1-family longitudinal block.
    "QB2": [0.95, 0.70, 0.20],  # QP replacement for a B2-family longitudinal block.
    "E0": [0.45, 0.45, 0.45],  # E0 end magnet color.
    "E1": [0.55, 0.55, 0.55],  # E1 end magnet color.
    "E2": [0.65, 0.65, 0.65],  # E2 end magnet color.
    "E3": [0.75, 0.75, 0.75],  # E3 end magnet color.
}

# DCC drawing body-block numbers replaced by shorter quasi-periodic magnets.
MERLIN_DEFAULT_QP_SHORT_BLOCKS = frozenset(
    {7, 13, 19, 25, 33, 39, 45, 51, 59, 65, 71, 77}
)

# Row placement and body-label pattern.
MERLIN_QUADRANTS = {  # Per-row placement, notch orientation, phase motion, and repeated body labels.
    "Q1": {  # Upper, +x row.
        "x_sign": 1.0,  # Row center is on the +x side of the horizontal gap.
        "z_sign": 1.0,  # Row center is on the +z side of the vertical gap.
        "shape_type": 2,  # MagnetBlock notch orientation type for this row.
        "phase_shifted": True,  # Q1 moves by the requested MERLIN phase.
        "body_pattern": (
            "B2",
            "A1",
            "B1",
            "A2",
        ),
    },
    "Q2": {  # Upper, -x row.
        "x_sign": -1.0,  # Row center is on the -x side of the horizontal gap.
        "z_sign": 1.0,  # Row center is on the +z side of the vertical gap.
        "shape_type": 1,  # MagnetBlock notch orientation type for this row.
        "phase_shifted": False,  # Q2 remains fixed when Q1/Q3 are phased.
        "body_pattern": (
            "B1",
            "A1",
            "B2",
            "A2",
        ),
    },
    "Q3": {  # Lower, -x row.
        "x_sign": -1.0,  # Row center is on the -x side of the horizontal gap.
        "z_sign": -1.0,  # Row center is on the -z side of the vertical gap.
        "shape_type": 1,  # MagnetBlock notch orientation type for this row.
        "phase_shifted": True,  # Q3 moves by the requested MERLIN phase.
        "body_pattern": (
            "B1",
            "A2",
            "B2",
            "A1",
        ),
    },
    "Q4": {  # Lower, +x row.
        "x_sign": 1.0,  # Row center is on the +x side of the horizontal gap.
        "z_sign": -1.0,  # Row center is on the -z side of the vertical gap.
        "shape_type": 2,  # MagnetBlock notch orientation type for this row.
        "phase_shifted": False,  # Q4 remains fixed when Q1/Q3 are phased.
        "body_pattern": (
            "B2",
            "A2",
            "B1",
            "A1",
        ),
    },
}

# E-type DCC end magnets decomposed from the vector detail drawings.
MERLIN_DCC_END_BLOCKS = {  # Upstream/downstream end pieces.
    "Q1": {  # Q1 end sequence.
        "upstream": [
            {
                "dcc_block": 2,
                "drawing_block_label": "2a",
                "type": "E3",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "down",
            },  # Outer narrow tip magnet.
            {
                "dcc_block": 2,
                "drawing_block_label": "2b",
                "type": "E3",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "right",
            },  # Inner narrow end marker before the long-span gap.
            {
                "dcc_block": 3,
                "drawing_block_label": "3",
                "type": "E3",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_AIR,
                "arrow": "up",
            },  # Inner E3 magnet in the long E span.
        ],
        "downstream": [
            {
                "dcc_block": 83,
                "drawing_block_label": "83",
                "type": "E0",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "up",
            },  # Inner end marker in the long E span.
            {
                "dcc_block": 85,
                "drawing_block_label": "85b",
                "type": "E0",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": 0.0,
                "arrow": "right",
            },  # Inner narrow end marker.
            {
                "dcc_block": 85,
                "drawing_block_label": "85a",
                "type": "E0",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "down",
            },  # Outer narrow end marker.
        ],
    },
    "Q2": {  # Q2 end sequence.
        "upstream": [
            {
                "dcc_block": 1,
                "drawing_block_label": "1a",
                "type": "E0",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "down",
            },  # Outer narrow tip magnet.
            {
                "dcc_block": 1,
                "drawing_block_label": "1b",
                "type": "E0",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "right",
            },  # Inner narrow E0 piece before the long-span gap.
            {
                "dcc_block": 3,
                "drawing_block_label": "3",
                "type": "E0",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_AIR,
                "arrow": "up",
            },  # Inner end marker in the long E span.
        ],
        "downstream": [
            {
                "dcc_block": 83,
                "drawing_block_label": "83",
                "type": "E3",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "up",
            },  # Inner end marker in the long E span.
            {
                "dcc_block": 84,
                "drawing_block_label": "84b",
                "type": "E3",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": 0.0,
                "arrow": "right",
            },  # Inner narrow end marker.
            {
                "dcc_block": 84,
                "drawing_block_label": "84a",
                "type": "E3",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "down",
            },  # Outer narrow end marker.
        ],
    },
    "Q3": {  # Q3 end sequence.
        "upstream": [
            {
                "dcc_block": 2,
                "drawing_block_label": "2a",
                "type": "E1",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "up",
            },  # Outer narrow tip magnet.
            {
                "dcc_block": 2,
                "drawing_block_label": "2b",
                "type": "E1",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "left",
            },  # Inner narrow end marker before the long-span gap.
            {
                "dcc_block": 3,
                "drawing_block_label": "3",
                "type": "E1",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_AIR,
                "arrow": "down",
            },  # Inner end marker in the long E span.
        ],
        "downstream": [
            {
                "dcc_block": 83,
                "drawing_block_label": "83",
                "type": "E2",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "down",
            },  # Inner end marker in the long E span.
            {
                "dcc_block": 85,
                "drawing_block_label": "85b",
                "type": "E2",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": 0.0,
                "arrow": "left",
            },  # Inner narrow end marker.
            {
                "dcc_block": 85,
                "drawing_block_label": "85a",
                "type": "E2",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "up",
            },  # Outer narrow end marker.
        ],
    },
    "Q4": {  # Q4 end sequence.
        "upstream": [
            {
                "dcc_block": 1,
                "drawing_block_label": "1a",
                "type": "E2",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "up",
            },  # Outer narrow tip magnet.
            {
                "dcc_block": 1,
                "drawing_block_label": "1b",
                "type": "E2",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "left",
            },  # Inner narrow end marker before the long-span gap.
            {
                "dcc_block": 3,
                "drawing_block_label": "3",
                "type": "E2",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_AIR,
                "arrow": "down",
            },  # Inner end marker in the long E span.
        ],
        "downstream": [
            {
                "dcc_block": 83,
                "drawing_block_label": "83",
                "type": "E1",
                "ly_mm": MERLIN_END_INNER_LY,
                "gap_after_mm": MERLIN_END_LONG_SPAN_GAP,
                "arrow": "down",
            },  # Inner end marker in the long E span.
            {
                "dcc_block": 84,
                "drawing_block_label": "84b",
                "type": "E1",
                "ly_mm": MERLIN_END_EDGE_INNER_LY,
                "gap_after_mm": 0.0,
                "arrow": "left",
            },  # Inner narrow end marker.
            {
                "dcc_block": 84,
                "drawing_block_label": "84a",
                "type": "E1",
                "ly_mm": MERLIN_END_EDGE_OUTER_LY,
                "gap_after_mm": 0.0,
                "arrow": "up",
            },  # Outer narrow end marker.
        ],
    },
}

MERLIN_BODY_DCC_FIRST = (
    4  # First DCC body-block number; this block is placed at phase - nper*period/2.
)
MERLIN_BODY_DCC_LAST = (
    MERLIN_BODY_DCC_FIRST + 4 * MERLIN_NPER - 2
)  # Last repeated DCC body-block number before the E-type end sequence.


def normalize_qp_short_blocks(qp_short_blocks=None):
    """Return validated DCC body-block numbers selected for QP shortening."""
    if qp_short_blocks is None:
        return set(MERLIN_DEFAULT_QP_SHORT_BLOCKS)

    normalized = set()
    try:
        iterator = iter(qp_short_blocks)
    except TypeError as exc:
        raise TypeError(
            "qp_short_blocks must be an iterable of integer DCC body-block indices"
        ) from exc

    for block in iterator:
        if isinstance(block, bool):
            raise TypeError(f"qp_short_blocks entries must be integers, got {block!r}")
        try:
            block = operator.index(block)
        except TypeError as exc:
            raise TypeError(
                f"qp_short_blocks entries must be integers, got {block!r}"
            ) from exc
        if not MERLIN_BODY_DCC_FIRST <= block <= MERLIN_BODY_DCC_LAST:
            raise ValueError(
                "qp_short_blocks entries must be DCC body-block indices "
                f"from {MERLIN_BODY_DCC_FIRST} to {MERLIN_BODY_DCC_LAST}, got {block}"
            )
        normalized.add(block)
    return normalized


def MagnetBlock(
    _pc,
    _wc,
    _cx,
    _cz,
    _type,
    _ndiv,
    _m,
    _top_cx=None,
    _top_cz=None,
    _top_slot_cap_lz=None,
    _mirror_z=False,
):
    # From RADIA_APPLE_II_Demo.py

    u = rad.ObjCnt([])

    top_cx = _cx if _top_cx is None else _top_cx
    top_cz = _cz if _top_cz is None else _top_cz
    top_z_sign = -1.0 if _mirror_z else 1.0
    lower_z_sign = -top_z_sign
    wwc = _wc
    if _type != 0:
        left_cx = top_cx if _type == 1 else _cx
        right_cx = top_cx if _type == 2 else _cx
        ppc = [
            _pc[0] + (left_cx - right_cx) / 2.0,
            _pc[1],
            _pc[2],
        ]
        wwc = [_wc[0] - left_cx - right_cx, _wc[1], _wc[2]]
    else:
        ppc = _pc

    b1 = rad.ObjRecMag(ppc, wwc, _m)
    rad.ObjAddToCnt(u, [b1])
    # ndiv2 = [1,_ndiv[1],_ndiv[2]]

    if (_cx > 0.01) and (_cz > 0.01):
        if _type == 1:
            b2 = []
            if _top_slot_cap_lz is None:
                ppc = [
                    _pc[0] - _wc[0] / 2 + top_cx / 2,
                    _pc[1],
                    _pc[2] - top_z_sign * top_cz / 2,
                ]
                wwc = [top_cx, _wc[1], _wc[2] - top_cz]
                b2.append(rad.ObjRecMag(ppc, wwc, _m))
            else:
                lower_lz = _wc[2] - _top_slot_cap_lz - top_cz
                ppc = [
                    _pc[0] - _wc[0] / 2 + top_cx / 2,
                    _pc[1],
                    _pc[2] - top_z_sign * (_top_slot_cap_lz + top_cz) / 2,
                ]
                wwc = [top_cx, _wc[1], lower_lz]
                b2.append(rad.ObjRecMag(ppc, wwc, _m))
                ppc = [
                    _pc[0] - _wc[0] / 2 + top_cx / 2,
                    _pc[1],
                    _pc[2] + top_z_sign * (_wc[2] - _top_slot_cap_lz) / 2,
                ]
                wwc = [top_cx, _wc[1], _top_slot_cap_lz]
                b2.append(rad.ObjRecMag(ppc, wwc, _m))

            ppc = [
                _pc[0] + _wc[0] / 2 - _cx / 2,
                _pc[1],
                _pc[2] - lower_z_sign * _cz / 2,
            ]
            wwc = [_cx, _wc[1], _wc[2] - _cz]
            b3 = rad.ObjRecMag(ppc, wwc, _m)
            rad.ObjAddToCnt(u, [*b2, b3])

        elif _type == 2:
            ppc = [
                _pc[0] - _wc[0] / 2 + _cx / 2,
                _pc[1],
                _pc[2] - lower_z_sign * _cz / 2,
            ]
            wwc = [_cx, _wc[1], _wc[2] - _cz]
            b2 = rad.ObjRecMag(ppc, wwc, _m)

            b3 = []
            if _top_slot_cap_lz is None:
                ppc = [
                    _pc[0] + _wc[0] / 2 - top_cx / 2,
                    _pc[1],
                    _pc[2] - top_z_sign * top_cz / 2,
                ]
                wwc = [top_cx, _wc[1], _wc[2] - top_cz]
                b3.append(rad.ObjRecMag(ppc, wwc, _m))
            else:
                lower_lz = _wc[2] - _top_slot_cap_lz - top_cz
                ppc = [
                    _pc[0] + _wc[0] / 2 - top_cx / 2,
                    _pc[1],
                    _pc[2] - top_z_sign * (_top_slot_cap_lz + top_cz) / 2,
                ]
                wwc = [top_cx, _wc[1], lower_lz]
                b3.append(rad.ObjRecMag(ppc, wwc, _m))
                ppc = [
                    _pc[0] + _wc[0] / 2 - top_cx / 2,
                    _pc[1],
                    _pc[2] + top_z_sign * (_wc[2] - _top_slot_cap_lz) / 2,
                ]
                wwc = [top_cx, _wc[1], _top_slot_cap_lz]
                b3.append(rad.ObjRecMag(ppc, wwc, _m))
            rad.ObjAddToCnt(u, [b2, *b3])

        elif _type == 3:
            ppc = [_pc[0] - _wc[0] / 2 + _cx / 2, _pc[1], _pc[2]]
            wwc = [_cx, _wc[1], _wc[2] - 2 * _cz]
            b2 = rad.ObjRecMag(ppc, wwc, _m)

            ppc = [_pc[0] + _wc[0] / 2 - _cx / 2, _pc[1], _pc[2]]
            wwc = [_cx, _wc[1], _wc[2] - 2 * _cz]
            b3 = rad.ObjRecMag(ppc, wwc, _m)
            rad.ObjAddToCnt(u, [b2, b3])

    rad.ObjDivMag(u, _ndiv, "Frame->LabTot")
    return u


def merlin_body_table(quadrant, phase=0.0, qp_short_blocks=None):
    """Return the known DCC body-magnet table for one MERLIN quadrant."""
    if quadrant not in MERLIN_QUADRANTS:
        raise ValueError(f"Unknown MERLIN quadrant: {quadrant}")

    cfg = MERLIN_QUADRANTS[quadrant]
    qp_short_blocks = normalize_qp_short_blocks(qp_short_blocks)
    entries = []

    for dcc_block in range(MERLIN_BODY_DCC_FIRST, MERLIN_BODY_DCC_LAST + 1):
        pattern_index = (dcc_block - MERLIN_BODY_DCC_FIRST) % 4
        base_type = cfg["body_pattern"][pattern_index]
        is_qp_short = dcc_block in qp_short_blocks
        block_type = f"Q{base_type}" if is_qp_short else base_type

        entries.append(
            {
                "dcc_block": dcc_block,
                "drawing_block_label": str(dcc_block),
                "base_type": base_type,
                "type": block_type,
                "is_qp_short": is_qp_short,
                "ly_mm": MERLIN_STD_LY,
                "arrow": MERLIN_MAGNET_TYPE_TO_ARROW[base_type],
            }
        )

    y_center = phase - MERLIN_NPER * MERLIN_PERIOD / 2.0
    for i, entry in enumerate(entries):
        if i > 0:
            y_center += (
                entries[i - 1]["ly_mm"] / 2.0 + MERLIN_AIR + entry["ly_mm"] / 2.0
            )
        entry["y_center_mm"] = y_center
        entry["body_block"] = (
            entry["dcc_block"] - (MERLIN_BODY_DCC_FIRST - 1)
            if MERLIN_BODY_DCC_FIRST <= entry["dcc_block"] <= MERLIN_BODY_DCC_LAST
            else None
        )
        local_m_unit = MERLIN_ARROW_TO_LOCAL_UNIT[entry["arrow"]]
        entry["m_unit"] = [
            local_m_unit[0],
            cfg["x_sign"] * cfg["z_sign"] * local_m_unit[1],
            cfg["z_sign"] * local_m_unit[2],
        ]

    return entries


def merlin_magnet_table(quadrant, phase=0.0, qp_short_blocks=None):
    """Return one quadrant table with known body magnets and E-end magnets."""
    if quadrant not in MERLIN_QUADRANTS:
        raise ValueError(f"Unknown MERLIN quadrant: {quadrant}")

    cfg = MERLIN_QUADRANTS[quadrant]
    row_phase = phase if cfg["phase_shifted"] else 0.0
    body_entries = [
        dict(entry, gap_after_mm=MERLIN_AIR)
        for entry in merlin_body_table(
            quadrant, phase=row_phase, qp_short_blocks=qp_short_blocks
        )
    ]
    entries = []

    entries.extend(dict(entry) for entry in MERLIN_DCC_END_BLOCKS[quadrant]["upstream"])
    entries.extend(body_entries)
    entries.extend(
        dict(entry) for entry in MERLIN_DCC_END_BLOCKS[quadrant]["downstream"]
    )

    block4_index = next(
        i
        for i, entry in enumerate(entries)
        if entry["dcc_block"] == MERLIN_BODY_DCC_FIRST
    )
    upstream_span_to_block4 = sum(
        entries[i]["ly_mm"] / 2.0
        + entries[i]["gap_after_mm"]
        + entries[i + 1]["ly_mm"] / 2.0
        for i in range(block4_index)
    )
    y_center = row_phase - MERLIN_NPER * MERLIN_PERIOD / 2.0 - upstream_span_to_block4

    for i, entry in enumerate(entries):
        if i > 0:
            y_center += (
                entries[i - 1]["ly_mm"] / 2.0
                + entries[i - 1]["gap_after_mm"]
                + entry["ly_mm"] / 2.0
            )
        entry["y_center_mm"] = y_center
        local_m_unit = MERLIN_ARROW_TO_LOCAL_UNIT[entry["arrow"]]
        if entry["type"].startswith("E"):
            entry["body_block"] = (
                None  # E labels are end magnets, not repeated body blocks.
            )
            entry["base_type"] = entry["type"]
            entry["is_qp_short"] = False
            # E-end detail arrows are already split by Q1/Q2 vs Q3/Q4, so do not add the body z-mirror to longitudinal arrows.
            entry["m_unit"] = [
                local_m_unit[0],
                cfg["x_sign"] * local_m_unit[1],
                cfg["z_sign"] * local_m_unit[2],
            ]
        else:
            entry["m_unit"] = [
                local_m_unit[0],
                cfg["x_sign"] * cfg["z_sign"] * local_m_unit[1],
                cfg["z_sign"] * local_m_unit[2],
            ]

    return entries


def MerlinMagnetArray(
    quadrant,
    phase=0.0,
    gap=15.0,
    qp_retraction=8.0,
    gapx=MERLIN_GAPX,
    br=MERLIN_BR,
    mu=MERLIN_MU,
    ndiv=MERLIN_NDIV,
    qp_short_blocks=None,
):
    """Build one MERLIN magnet row from known body magnets plus E ends."""
    if quadrant not in MERLIN_QUADRANTS:
        raise ValueError(f"Unknown MERLIN quadrant: {quadrant}")

    cfg = MERLIN_QUADRANTS[quadrant]
    row = rad.ObjCnt([])
    px = cfg["x_sign"] * (MERLIN_STD_LX / 2.0 + gapx / 2.0)
    pz = cfg["z_sign"] * (gap / 2.0 + MERLIN_STD_LZ / 2.0)

    for entry in merlin_magnet_table(
        quadrant, phase=phase, qp_short_blocks=qp_short_blocks
    ):
        pc = [px, entry["y_center_mm"], pz]
        wc = [MERLIN_STD_LX, entry["ly_mm"], MERLIN_STD_LZ]

        if entry["is_qp_short"]:
            # Shorten QP blocks on the gap side while preserving the outer face.
            qp_lz = MERLIN_STD_LZ - qp_retraction
            wc[2] = qp_lz
            pc[2] += cfg["z_sign"] * qp_retraction / 2.0

        m = [br * component for component in entry["m_unit"]]
        block = MagnetBlock(
            pc,
            wc,
            MERLIN_CX,
            MERLIN_CZ,
            cfg["shape_type"],
            ndiv,
            m,
            MERLIN_TOP_CX,
            MERLIN_TOP_CZ,
            None if entry["is_qp_short"] else MERLIN_TOP_SLOT_CAP_LZ,
            cfg["z_sign"] > 0,
        )
        rad.ObjDrwAtr(block, MERLIN_MAGNET_COLORS[entry["type"]], 0.0001)
        rad.ObjAddToCnt(row, [block])

    mat = rad.MatLin(mu, abs(br))
    rad.MatApl(row, mat)
    return row


def MERLIN_APPLE_II(
    phase=0.0,
    gap=15.0,
    qp_retraction=8.0,
    gapx=MERLIN_GAPX,
    br=MERLIN_BR,
    mu=MERLIN_MU,
    ndiv=MERLIN_NDIV,
    qp_short_blocks=None,
):
    """Build the full four-row MERLIN QP-EPU magnet-only model.

    The input phase is the MERLIN measurement phase in millimeters: Q1 and Q3
    are shifted by this amount, Q2 and Q4 remain fixed.
    """
    qp_short_blocks = normalize_qp_short_blocks(qp_short_blocks)
    q1 = MerlinMagnetArray(
        "Q1",
        phase=phase,
        gap=gap,
        qp_retraction=qp_retraction,
        gapx=gapx,
        br=br,
        mu=mu,
        ndiv=ndiv,
        qp_short_blocks=qp_short_blocks,
    )
    q2 = MerlinMagnetArray(
        "Q2",
        phase=phase,
        gap=gap,
        qp_retraction=qp_retraction,
        gapx=gapx,
        br=br,
        mu=mu,
        ndiv=ndiv,
        qp_short_blocks=qp_short_blocks,
    )
    q3 = MerlinMagnetArray(
        "Q3",
        phase=phase,
        gap=gap,
        qp_retraction=qp_retraction,
        gapx=gapx,
        br=br,
        mu=mu,
        ndiv=ndiv,
        qp_short_blocks=qp_short_blocks,
    )
    q4 = MerlinMagnetArray(
        "Q4",
        phase=phase,
        gap=gap,
        qp_retraction=qp_retraction,
        gapx=gapx,
        br=br,
        mu=mu,
        ndiv=ndiv,
        qp_short_blocks=qp_short_blocks,
    )
    rows = {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4}
    return rad.ObjCnt([q1, q2, q3, q4]), rows


def build_merlin(
    gap=15.0,
    z=0.0,
    mode="parallel",
    qp_retraction=8.0,
    relax: bool = False,
    qp_short_blocks=None,
):
    """Build the full MERLIN magnet-only model with the specified gap and z."""
    if mode == "parallel":
        rad.UtiDelAll()
        merlin, _ = MERLIN_APPLE_II(
            phase=-z,
            gap=gap,
            qp_retraction=qp_retraction,
            qp_short_blocks=qp_short_blocks,
        )
    else:
        # TODO: implement antiparallel movement
        raise NotImplementedError()

    if relax:
        # For nonlinear/hybrid structures, relax the model before sampling the field.
        intrc = rad.RlxPre(merlin)
        rad.RlxAuto(intrc, 1e-4, 1000)

    return merlin
