"""Extract field values from the RADIA model and translate to SRW."""

import pathlib
from array import array

import numpy as np
from srwpy.srwlib import SRWLMagFld3D, SRWLMagFldC, srwl_uti_read_mag_fld_3d

from correction import correct_one_field_component
from model import (
    MERLIN_DEFAULT_QP_SHORT_BLOCKS,
    build_merlin,
    normalize_qp_short_blocks,
)
from reference import load_merlin_hall_scan

FIELD_CACHE_DIR = pathlib.Path("fields")
FIELD_CACHE_DIR.mkdir(exist_ok=True)


def _sample_field(
    rad_obj,
    x_grid,
    y_grid,
    z_grid,
    *,
    correct_field_integral: bool = False,
    correction_kick_length_m: float = 1.8925,
    correction_kick_rms_len_mm: float = 100.0,
):
    import radia as rad

    # Build SRW-grid points with shape (nz, ny, nx).
    Z_s, Y_s, X_s = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")

    # Convert SRW coordinates [m] to RADIA coordinates [mm]. Normally, we would just add
    # a - sign to Y (the Z in SRW), but our model happens to be flipped in X.
    points_radia_mm = np.column_stack(
        [
            -1e3 * X_s.ravel(order="C"),
            1e3 * Z_s.ravel(order="C"),
            -1e3 * Y_s.ravel(order="C"),
        ]
    )

    # One RADIA call for all observer points.
    # Per your RADIA docstring, the empty component string returns [Bx, By, Bz].
    b_radia = np.asarray(rad.Fld(rad_obj, "b", points_radia_mm.tolist()), dtype=float)
    if b_radia.shape != (points_radia_mm.shape[0], 3):
        raise ValueError(f"Unexpected RADIA field shape: {b_radia.shape}")

    # Convert RADIA field components to SRW field components.
    bx_srw = -b_radia[:, 0]
    by_srw = -b_radia[:, 2]
    bz_srw = b_radia[:, 1]

    if correct_field_integral:
        bx_srw, _ = correct_one_field_component(
            z_grid,
            bx_srw,
            dist_between_kicks_m=correction_kick_length_m,
            rms_len_kick_mm=correction_kick_rms_len_mm,
        )

        by_srw, _ = correct_one_field_component(
            z_grid,
            by_srw,
            dist_between_kicks_m=correction_kick_length_m,
            rms_len_kick_mm=correction_kick_rms_len_mm,
        )

    # SRW expects Python array('d') for tabulated fields.
    arBx = array("d", bx_srw)
    arBy = array("d", by_srw)
    arBz = array("d", bz_srw)
    print(f"Sampled {len(arBx):,} field points")

    return arBx, arBy, arBz


def _make_field_cache_key(
    epu_gap_mm: float,
    epu_z_mm: float,
    epu_mode: str,
    qp_retraction: float,
    x_hw: float,
    y_hw: float,
    z_hw: float,
    nx: int,
    ny: int,
    nz: int,
    correct_field_integral: bool,
    correction_kick_length_m: float,
    correction_kick_rms_len_mm: float,
    qp_short_blocks=None,
):
    key = (
        f"gap{epu_gap_mm:.3g}_z{epu_z_mm:.3g}_{epu_mode}"
        f"_xhw{x_hw:.3g}_yhw{y_hw:.3g}_zhw{z_hw:.3g}"
        f"_nx{nx}_ny{ny}_nz{nz}"
        f"_corr{correct_field_integral}"
    )
    if qp_retraction != 8.0:
        key += f"_qpret{qp_retraction:.3g}"

    if qp_short_blocks is not None:
        normalized_qp_short_blocks = normalize_qp_short_blocks(qp_short_blocks)
        if normalized_qp_short_blocks != set(MERLIN_DEFAULT_QP_SHORT_BLOCKS):
            if normalized_qp_short_blocks:
                joined_blocks = "-".join(
                    str(i) for i in sorted(normalized_qp_short_blocks)
                )
                key += f"_qpshort{joined_blocks}"
            else:
                key += "_qpshortnone"

    if correct_field_integral:
        key += (
            f"_kicklen{correction_kick_length_m:.3g}"
            f"_kickrms{correction_kick_rms_len_mm:.3g}"
        )
    return key + ".dat"


def make_field(
    epu_gap: float = 15.0,
    epu_z: float = 0.0,
    epu_mode="parallel",
    qp_retraction=8.0,
    x_hw: float = 0.0,
    y_hw: float = 0.0,
    z_hw: float = 1.1,
    nx: int = 1,
    ny: int = 1,
    nz: int = 2201,
    correct_field_integral: bool = False,
    correction_kick_length_m: float = 1.8925,
    correction_kick_rms_len_mm: float = 100.0,
    write_cache: bool = True,
    qp_short_blocks=None,
) -> SRWLMagFldC:
    if qp_short_blocks is not None:
        qp_short_blocks = normalize_qp_short_blocks(qp_short_blocks)

    cache_key = _make_field_cache_key(
        epu_gap_mm=epu_gap,
        epu_z_mm=epu_z,
        epu_mode=epu_mode,
        qp_retraction=qp_retraction,
        qp_short_blocks=qp_short_blocks,
        x_hw=x_hw,
        y_hw=y_hw,
        z_hw=z_hw,
        nx=nx,
        ny=ny,
        nz=nz,
        correct_field_integral=correct_field_integral,
        correction_kick_length_m=correction_kick_length_m,
        correction_kick_rms_len_mm=correction_kick_rms_len_mm,
    )
    print(f"Field cache key: {cache_key}")
    field_cache_file = (FIELD_CACHE_DIR / cache_key).resolve()
    if not field_cache_file.exists():
        rad_obj = build_merlin(
            gap=epu_gap,
            z=epu_z,
            mode=epu_mode,
            qp_retraction=qp_retraction,
            qp_short_blocks=qp_short_blocks,
        )
        x_grid = np.linspace(-x_hw, x_hw, nx)
        y_grid = np.linspace(-y_hw, y_hw, ny)
        z_grid = np.linspace(-z_hw, z_hw, nz)

        arBx, arBy, arBz = _sample_field(
            rad_obj,
            x_grid,
            y_grid,
            z_grid,
            correct_field_integral=correct_field_integral,
            correction_kick_length_m=correction_kick_length_m,
            correction_kick_rms_len_mm=correction_kick_rms_len_mm,
        )

        rx = 0.0 if nx == 1 else float(x_grid[-1] - x_grid[0])
        ry = 0.0 if ny == 1 else float(y_grid[-1] - y_grid[0])
        rz = float(z_grid[-1] - z_grid[0])

        x_center = 0.5 * float(x_grid[0] + x_grid[-1])
        y_center = 0.5 * float(y_grid[0] + y_grid[-1])
        z_center = 0.5 * float(z_grid[0] + z_grid[-1])

        mag3d = SRWLMagFld3D(arBx, arBy, arBz, nx, ny, nz, rx, ry, rz, 1)
        if write_cache:
            mag3d.save_ascii(
                str(field_cache_file), _xc=x_center, _yc=y_center, _zc=z_center
            )
        mag_cnt = SRWLMagFldC(
            [mag3d],
            array("d", [x_center]),
            array("d", [y_center]),
            array("d", [z_center]),
        )
        print("SRW magnetic field object created")
        print(f"grid: nx={nx}, ny={ny}, nz={nz}")
        print(f"ranges [m]: rx={rx:g}, ry={ry:g}, rz={rz:g}")
    else:
        try:
            mag_cnt = srwl_uti_read_mag_fld_3d(str(field_cache_file))
        except Exception:
            print(f"Failed to load field cache from {field_cache_file}, recomputing")
            field_cache_file.unlink()
            return make_field(
                epu_gap=epu_gap,
                epu_z=epu_z,
                epu_mode=epu_mode,
                qp_retraction=qp_retraction,
                x_hw=x_hw,
                y_hw=y_hw,
                z_hw=z_hw,
                nx=nx,
                ny=ny,
                nz=nz,
                correct_field_integral=correct_field_integral,
                correction_kick_length_m=correction_kick_length_m,
                correction_kick_rms_len_mm=correction_kick_rms_len_mm,
                write_cache=write_cache,
                qp_short_blocks=qp_short_blocks,
            )
        print(f"Loaded field from cache: {field_cache_file}")

    return mag_cnt


def make_field_from_file(
    file_path: str,
    z_hw: float = 1.1,
    nz: int = 2201,
    correct_field_integral: bool = False,
    correction_kick_length_m: float = 1.8925,
    correction_kick_rms_len_mm: float = 100.0,
) -> SRWLMagFldC:
    """Build a field container from a Hall-probe scan file."""
    z_grid = np.linspace(-z_hw, z_hw, nz)
    ds = load_merlin_hall_scan(file_path).drop_duplicates("z_mm").sortby("z_mm")
    ds = ds.interp(z_mm=z_grid * 1e3, assume_sorted=True).fillna(0.0)

    bx_srw = ds["bx_t"].values
    by_srw = ds["by_t"].values

    if correct_field_integral:
        bx_srw, _ = correct_one_field_component(
            z_grid,
            bx_srw,
            dist_between_kicks_m=correction_kick_length_m,
            rms_len_kick_mm=correction_kick_rms_len_mm,
        )
        by_srw, _ = correct_one_field_component(
            z_grid,
            by_srw,
            dist_between_kicks_m=correction_kick_length_m,
            rms_len_kick_mm=correction_kick_rms_len_mm,
        )

    arBx = array("d", bx_srw)
    arBy = array("d", by_srw)
    arBz = array("d", np.zeros_like(bx_srw))

    mag3d = SRWLMagFld3D(arBx, arBy, arBz, 1, 1, nz, 0.0, 0.0, 2 * z_hw, 1)
    return SRWLMagFldC(
        [mag3d],
        array("d", [0.0]),
        array("d", [0.0]),
        array("d", [0.0]),
    )
