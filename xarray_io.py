import pathlib

import numpy as np
import json
import xarray as xr
from srwpy.srwlib import srwl_uti_read_intens_hdf5


def _decode_hdf5_attr(value):
    """Decode h5py/SRW attrs that may be bytes, numpy bytes, or arrays."""
    if value is None:
        return None

    if isinstance(value, bytes):
        return value.decode("utf-8")

    arr = np.asarray(value)

    if arr.ndim == 0:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item

    out = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def srw_hdf5_output_to_xarray(ar_int, mesh, meta, *, drop_single_stokes=True):
    """
    Convert the outputs of srwl_uti_read_intens_hdf5(...) to an xarray.DataArray.

    Expected input:
        ar_int, mesh, meta, wfr = srwl_uti_read_intens_hdf5(path)

    """
    n_stokes = int(meta.get("n_stokes", 1))
    mutual = int(meta.get("mutual", 0))
    cmplx = int(meta.get("cmplx", 0))

    if mutual != 0:
        raise NotImplementedError(
            "This helper is for ordinary intensity/Stokes meshes, not mutual intensity/CSD output."
        )

    if cmplx != 0:
        raise NotImplementedError(
            "This helper assumes real intensity/Stokes data, i.e. meta['cmplx'] == 0."
        )

    ne = int(mesh.ne)
    nx = int(mesh.nx)
    ny = int(mesh.ny)

    energy = np.linspace(float(mesh.eStart), float(mesh.eFin), ne)
    x = np.linspace(float(mesh.xStart), float(mesh.xFin), nx)
    y = np.linspace(float(mesh.yStart), float(mesh.yFin), ny)

    raw = np.asarray(ar_int)

    expected = n_stokes * ne * nx * ny
    if raw.size != expected:
        raise ValueError(
            f"Unexpected SRW array size: got {raw.size}, expected {expected} "
            f"from n_stokes={n_stokes}, ne={ne}, nx={nx}, ny={ny}."
        )

    # SRW flat order is C-aligned:
    #   inner loop = photon energy
    #   then x
    #   outer loop = y
    #
    # With Stokes/components:
    #   component, y, x, energy
    arr = raw.reshape(n_stokes, ny, nx, ne).transpose(0, 3, 1, 2)

    labels = _decode_hdf5_attr(meta.get("arLabels"))
    units = _decode_hdf5_attr(meta.get("arUnits"))

    attrs = {
        "srw_n_stokes": n_stokes,
        "srw_mutual": mutual,
        "srw_cmplx": cmplx,
    }

    if labels is not None:
        attrs["srw_labels"] = labels
    if units is not None:
        attrs["srw_units"] = units
        if len(units) >= 4:
            attrs["units"] = units[3]

    if n_stokes == 1 and drop_single_stokes:
        da = xr.DataArray(
            arr[0],
            dims=("hv", "y", "x"),
            coords={
                "hv": ("hv", energy, {"units": "eV"}),
                "y": ("y", y * 1e3, {"units": "mm"}),
                "x": ("x", x * 1e3, {"units": "mm"}),
            },
            attrs=attrs,
        )
    else:
        stokes_labels = np.arange(n_stokes, dtype=np.int64)

        da = xr.DataArray(
            arr,
            dims=("stokes", "hv", "y", "x"),
            coords={
                "stokes": stokes_labels,
                "hv": ("hv", energy, {"units": "eV"}),
                "y": ("y", y * 1e3, {"units": "mm"}),
                "x": ("x", x * 1e3, {"units": "mm"}),
            },
            attrs=attrs,
        )

    return da


def convert_files_in_folder(folder_path):
    """Convert all SRW HDF5 output files in the given folder to xarray."""
    folder = pathlib.Path(folder_path)
    for h5_file in folder.glob("*.h5"):
        if not h5_file.stem.endswith("_xarray"):
            ar_int, mesh, meta, _ = srwl_uti_read_intens_hdf5(str(h5_file))
            da = srw_hdf5_output_to_xarray(ar_int, mesh, meta).rename(h5_file.stem)

            meta = h5_file.with_stem(h5_file.stem + "_info").with_suffix(".json")
            if meta.is_file():
                with meta.open("r") as f:
                    extra_attrs = json.load(f)
                da = da.assign_attrs(**extra_attrs)
            da.to_netcdf(
                h5_file.with_stem(h5_file.stem + "_xarray"),
                engine="h5netcdf",
                invalid_netcdf=True,
            )
