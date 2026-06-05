import pathlib

import numpy as np
import xarray as xr

UNDULATOR_CENTER_COUNT1 = 1045.9  # This corresponds to model 0


def load_merlin_hall_scan(file_path) -> xr.Dataset:
    """Load one MERLIN Hall-probe scan as a lean xarray Dataset."""
    path = pathlib.Path(file_path)
    header = {}
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[-2:] == ["EPU", "Gap"]:
            header["gap_mm"] = float(parts[0])
        elif len(parts) >= 4 and parts[-3:] == ["EPU", "Row", "Phase"]:
            header["phase_mm"] = float(parts[0])
        elif len(parts) == 5 and parts[0].isdigit():
            rows.append([float(value) for value in parts])

    data = np.asarray(rows, dtype=float)
    if data.size == 0:
        raise ValueError(f"No Hall-probe rows found in {path}")

    point = data[:, 0].astype(int)
    count_1_mm = data[:, 3]
    return (
        xr.Dataset(
            data_vars={
                "bx_t": ("point", data[:, 1] / 5.0),
                "by_t": ("point", data[:, 2] / 5.0),
            },
            coords={
                "point": point,
                "filename": path.name,
                "gap_mm": header.get("gap_mm"),
                "phase_mm": header.get("phase_mm"),
                "z_mm": ("point", count_1_mm - UNDULATOR_CENTER_COUNT1),
            },
        )
        .swap_dims({"point": "z_mm"})
        .drop_vars("point")
    )
