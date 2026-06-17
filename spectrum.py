"""Single-electron trajectory calculation."""

import array
import datetime
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import xarray as xr
from mpi4py import MPI
from srwpy.srwlib import (
    SRWLMagFldC,
    SRWLPartBeam,
    SRWLRadMesh,
    SRWLWfr,
    SRWLOptA,
    SRWLOptC,
    SRWLOptD,
    SRWLOptG,
    SRWLOptMirPl,
    srwl,
    srwl_wfr_emit_prop_multi_e,
)
from beam import get_beam_params


DEFAULT_PROPAGATION_PARAMS = [
    0,
    0,
    1.0,
    1,
    0,
    1.0,
    1.0,
    1.0,
    1.0,
    0,
    0,
    0,
]


def build_opt_bl_from_config(config: dict):
    """Build an SRW optics container from a JSON-serializable config."""
    if not isinstance(config, dict):
        raise TypeError("opt_bl_config must be a dict.")
    kind = config.get("kind", "vlspgm")
    if kind != "vlspgm":
        raise ValueError(f"Unsupported opt_bl_config kind: {kind!r}")
    return build_vlspgm_opt_bl(**{k: v for k, v in config.items() if k != "kind"})


def _xarray_safe_attrs(attrs: dict) -> dict:
    """Return attrs that xarray can serialize to netCDF/HDF5."""
    safe_attrs = {}
    for key, value in attrs.items():
        if isinstance(value, dict):
            safe_attrs[key] = json.dumps(value, sort_keys=True)
        else:
            safe_attrs[key] = value
    return safe_attrs


def _singleton_spatial_width_mm(dim: str, attrs: dict) -> float:
    opt_bl_config = attrs.get("opt_bl_config") or {}
    if not isinstance(opt_bl_config, dict):
        opt_bl_config = {}

    dispersion_plane = opt_bl_config.get("dispersion_plane", "vertical")
    exit_slit_width_m = opt_bl_config.get("exit_slit_width_m")
    non_disp_slit_size_m = opt_bl_config.get("non_disp_slit_size_m", 5e-3)

    if dispersion_plane == "vertical":
        if dim == "x" and non_disp_slit_size_m is not None:
            return float(non_disp_slit_size_m) * 1e3
        if dim == "y" and exit_slit_width_m is not None:
            return float(exit_slit_width_m) * 1e3
    elif dispersion_plane == "horizontal":
        if dim == "y" and non_disp_slit_size_m is not None:
            return float(non_disp_slit_size_m) * 1e3
        if dim == "x" and exit_slit_width_m is not None:
            return float(exit_slit_width_m) * 1e3

    range_m = attrs.get(f"{dim}_range_m")
    if range_m is not None:
        return float(range_m) * 1e3

    obs_z_m = attrs.get("obs_z_m")
    theta_half_mrad = attrs.get(f"theta_{dim}_half_mrad")
    if obs_z_m is not None and theta_half_mrad is not None:
        return 2.0 * float(obs_z_m) * float(theta_half_mrad)

    raise ValueError(f"Cannot infer spatial integration width for singleton {dim!r}.")


def _integrate_srw_intensity_mesh(data: xr.DataArray, attrs: dict) -> xr.DataArray:
    """Integrate an SRW intensity mesh over x/y to estimate total flux."""
    original_attrs = dict(data.attrs)
    out = data
    for dim in ("x", "y"):
        if dim not in out.dims:
            continue
        if out.sizes[dim] > 1:
            out = out.integrate(dim)
        else:
            out = out.squeeze(dim, drop=True) * _singleton_spatial_width_mm(dim, attrs)

    units = original_attrs.get("units")
    if isinstance(units, str) and "/mm^2" in units:
        original_attrs["units"] = units.replace("/mm^2", "")

    srw_units = original_attrs.get("srw_units")
    if isinstance(srw_units, list) and len(srw_units) >= 4:
        units = original_attrs.get("units")
        if isinstance(units, str):
            srw_units = list(srw_units)
            srw_units[3] = units
            original_attrs["srw_units"] = srw_units

    original_attrs["spatially_integrated"] = True
    original_attrs["spatial_integration_note"] = (
        "Integrated propagated SRW intensity over x/y after readback."
    )
    return out.assign_attrs(**original_attrs)


def build_vlspgm_opt_bl(
    *,
    set_energy_eV: float,
    g0: float,
    r_prime_m: float,
    cff0: float,
    g1: float = 0.0,
    g2: float = 0.0,
    g3: float = 0.0,
    g4: float = 0.0,
    order: int = 1,
    grating_length_m: float = 0.12,
    grating_width_m: float = 0.02,
    exit_slit_width_m: float | None = None,
    non_disp_slit_size_m: float = 5e-3,
    dispersion_plane: str = "vertical",
    reflectivity: float = 1.0,
    propagation_params: list[float] | None = None,
):
    """Build a minimal VLS-PGM beamline: grating, drift, optional exit slit."""
    if dispersion_plane not in {"vertical", "horizontal"}:
        raise ValueError("dispersion_plane must be 'vertical' or 'horizontal'.")

    pp = (
        DEFAULT_PROPAGATION_PARAMS if propagation_params is None else propagation_params
    )
    align_energy_eV = order * set_energy_eV
    ang_roll = 0.0 if dispersion_plane == "vertical" else 0.5 * np.pi

    substrate = SRWLOptMirPl(
        _size_tang=grating_length_m,
        _size_sag=grating_width_m,
        _refl=reflectivity,
    )
    grating = SRWLOptG(
        substrate,
        _m=order,
        _grDen=g0,
        _grDen1=g1,
        _grDen2=g2,
        _grDen3=g3,
        _grDen4=g4,
        _e_avg=align_energy_eV,
        _cff=cff0,
        _ang_roll=ang_roll,
    )

    elements = [grating, SRWLOptD(r_prime_m)]
    if exit_slit_width_m is not None:
        if dispersion_plane == "vertical":
            slit = SRWLOptA("r", "a", _Dx=non_disp_slit_size_m, _Dy=exit_slit_width_m)
        else:
            slit = SRWLOptA("r", "a", _Dx=exit_slit_width_m, _Dy=non_disp_slit_size_m)
        elements.append(slit)

    return SRWLOptC(elements, [list(pp) for _ in range(len(elements) + 1)])


def calculate_spectrum_multi_mpi(
    field_container: SRWLMagFldC,
    file_path: str | pathlib.Path,
    *,
    n_processes: int = 8,
    mpi_executable: str | pathlib.Path = "mpirun",
    overwrite: bool = True,
    return_xarray: bool = True,
    use_mpi: bool = True,
    only_flux: bool = False,
    **spectrum_kwargs,
):
    """Run calculate_spectrum_multi through mpirun and block until it finishes.

    Set use_mpi=False to run in-process while keeping the same output readback.
    If return_xarray=True, the HDF5 output is read back with
    srwl_uti_read_intens_hdf5() and srw_hdf5_output_to_xarray(), saved next to
    the SRW output as *_xarray.h5, then returned.
    """
    if use_mpi and n_processes < 1:
        raise ValueError("n_processes must be at least 1.")

    if use_mpi and MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError(
            "calculate_spectrum_multi_mpi() should be called from a normal Python "
            "process, not from inside an existing MPI worker."
        )

    file_path = pathlib.Path(file_path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = file_path.with_name(file_path.stem + "_info.json")
    xarray_path = file_path.with_name(file_path.stem + "_xarray" + file_path.suffix)

    if file_path.exists() and not overwrite:
        raise FileExistsError(f"{file_path} already exists; pass overwrite=True.")
    if return_xarray and xarray_path.exists() and not overwrite:
        raise FileExistsError(f"{xarray_path} already exists; pass overwrite=True.")

    if overwrite:
        file_path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)
        xarray_path.unlink(missing_ok=True)

    spectrum_kwargs = dict(spectrum_kwargs)
    spectrum_kwargs.pop("field_container", None)
    spectrum_kwargs.pop("file_path", None)
    spectrum_kwargs.pop("use_mpi", None)
    spectrum_kwargs["only_flux"] = only_flux
    opt_bl_config = spectrum_kwargs.get("opt_bl_config")
    if opt_bl_config is not None:
        # Fail here, before mpirun, if the config is not worker-serializable.
        json.dumps(opt_bl_config)
        extra_attrs = dict(spectrum_kwargs.get("extra_attrs") or {})
        extra_attrs.setdefault("opt_bl_config", opt_bl_config)
        spectrum_kwargs["extra_attrs"] = extra_attrs
    if use_mpi and spectrum_kwargs.get("opt_bl") is not None:
        raise ValueError(
            "calculate_spectrum_multi_mpi(use_mpi=True) cannot serialize opt_bl. "
            "Use use_mpi=False, or pass a serializable optics config and build "
            "the SRWLOptC inside the worker."
        )

    if use_mpi:
        with tempfile.TemporaryDirectory(
            prefix=f"{file_path.stem}_mpi_", dir=file_path.parent
        ) as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)
            field_path = tmp_dir / "field.dat"
            config_path = tmp_dir / "config.json"

            if len(field_container.arMagFld) != 1:
                raise NotImplementedError(
                    "calculate_spectrum_multi_mpi() supports one SRW 3D field."
                )

            field = field_container.arMagFld[0]
            field.save_ascii(
                str(field_path),
                _xc=float(field_container.arXc[0])
                if len(field_container.arXc)
                else 0.0,
                _yc=float(field_container.arYc[0])
                if len(field_container.arYc)
                else 0.0,
                _zc=float(field_container.arZc[0])
                if len(field_container.arZc)
                else 0.0,
            )

            config = {
                "field_path": str(field_path),
                "file_path": str(file_path),
                "spectrum_kwargs": spectrum_kwargs,
            }
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)

            worker_code = "\n".join(
                [
                    "import json, sys",
                    "from srwpy.srwlib import srwl_uti_read_mag_fld_3d",
                    "from spectrum import build_opt_bl_from_config, calculate_spectrum_multi",
                    "with open(sys.argv[1], 'r') as f:",
                    "    config = json.load(f)",
                    "field_container = srwl_uti_read_mag_fld_3d(config['field_path'])",
                    "kwargs = dict(config.get('spectrum_kwargs', {}))",
                    "kwargs.pop('use_mpi', None)",
                    "opt_bl_config = kwargs.pop('opt_bl_config', None)",
                    "if opt_bl_config is not None:",
                    "    kwargs['opt_bl'] = build_opt_bl_from_config(opt_bl_config)",
                    "calculate_spectrum_multi(",
                    "    field_container=field_container,",
                    "    file_path=config['file_path'],",
                    "    use_mpi=True,",
                    "    **kwargs,",
                    ")",
                ]
            )

            command = [
                str(mpi_executable),
                "-np",
                str(n_processes),
                sys.executable,
                "-c",
                worker_code,
                str(config_path),
            ]

            subprocess.run(
                command,
                check=True,
                cwd=pathlib.Path(__file__).resolve().parent,
            )
    else:
        opt_bl_config = spectrum_kwargs.pop("opt_bl_config", None)
        if opt_bl_config is not None:
            if spectrum_kwargs.get("opt_bl") is not None:
                raise ValueError("Pass either opt_bl or opt_bl_config, not both.")
            spectrum_kwargs["opt_bl"] = build_opt_bl_from_config(opt_bl_config)
        calculate_spectrum_multi(
            field_container=field_container,
            file_path=str(file_path),
            use_mpi=False,
            **spectrum_kwargs,
        )

    if not file_path.exists():
        raise FileNotFoundError(f"Spectrum calculation did not create {file_path}")

    if not return_xarray:
        return file_path

    from srwpy.srwlib import srwl_uti_read_intens_hdf5
    from xarray_io import srw_hdf5_output_to_xarray

    ar_int, mesh, meta, _ = srwl_uti_read_intens_hdf5(str(file_path))
    data = srw_hdf5_output_to_xarray(ar_int, mesh, meta).rename(file_path.stem)
    info_attrs = {}
    if info_path.is_file():
        with info_path.open("r") as f:
            info_attrs = json.load(f)
            data = data.assign_attrs(**_xarray_safe_attrs(info_attrs))
    if info_attrs.get("only_flux") and info_attrs.get("has_opt_bl"):
        data = _integrate_srw_intensity_mesh(data, info_attrs)
    data = data.squeeze()
    data.to_netcdf(xarray_path, engine="h5netcdf", invalid_netcdf=True)
    return data


def calculate_spectrum_single(
    field_container: SRWLMagFldC,
    *,
    obs_z_m: float = 10.0,  # Observation point z in meters
    x0_m: float = 0.0,  # Initial x position in meters
    y0_m: float = 0.0,  # Initial y position in meters
    z0_m: float | None = None,  # Initial z position, defaults to the start of the field
    xp0_rad: float = 0.0,  # Instant transverse velocity x'/c (angle for relativistic particles)
    yp0_rad: float = 0.0,  # Instant transverse velocity y'/c (angle for relativistic particles)
    hv_start_eV: float = 1.0,  # Start photon energy in eV
    hv_end_eV: float = 100.0,  # End photon energy in eV
    nhv: int = 5001,
    nx: int = 1,
    ny: int = 1,
    x_range_m: float = 0.01,  # Total horizontal range at the observation plane in meters
    y_range_m: float = 0.01,  # Total vertical range at the observation plane in meters
    beam: str = "ALS",
):
    """Calculate a single-electron trajectory through the given field."""

    # Get start Z assuming symmetric single field
    z_hw = field_container.arMagFld[0].rz / 2
    if z0_m is None:
        # Assuming symmetric field around z=0, start at the beginning of the field
        z0_m = -z_hw

    # Setup single-electron beam
    iavg, e, _, _, _, _, _ = get_beam_params(beam)
    elec_beam = SRWLPartBeam()
    elec_beam.Iavg = iavg

    elec_beam.partStatMom1.x = x0_m
    elec_beam.partStatMom1.y = y0_m
    elec_beam.partStatMom1.z = z0_m
    elec_beam.partStatMom1.xp = xp0_rad
    elec_beam.partStatMom1.yp = yp0_rad
    elec_beam.partStatMom1.gamma = e / 0.510998950e-3
    elec_beam.partStatMom1.relE0 = 1
    elec_beam.partStatMom1.nq = -1

    wfr = SRWLWfr()

    wfr.allocate(nhv, nx, ny)
    wfr.mesh.eStart = hv_start_eV
    wfr.mesh.eFin = hv_end_eV
    wfr.mesh.zStart = obs_z_m
    wfr.mesh.xStart = (-0.5 * x_range_m) if nx > 1 else 0.0
    wfr.mesh.xFin = (0.5 * x_range_m) if nx > 1 else 0.0
    wfr.mesh.yStart = (-0.5 * y_range_m) if ny > 1 else 0.0
    wfr.mesh.yFin = (0.5 * y_range_m) if ny > 1 else 0.0
    wfr.partBeam = elec_beam

    # Manual integration over the finite field-map length.
    sr_prec = [
        0,  # manual method
        0.001,  # relative precision
        -z_hw,  # start integration [m]
        z_hw,  # end integration [m]
        10000,  # trajectory points for SR integral
        0,  # use terminating terms
        0,  # sampling factor
    ]
    srwl.CalcElecFieldSR(wfr, 0, field_container, sr_prec)

    intensity = array.array("f", [0.0] * (nhv * nx * ny))
    srwl.CalcIntFromElecField(
        intensity,
        wfr,
        6,  # total polarization
        0,  # 0 is single-electron, 1 is multi-electron intensity
        6,
        0,  # inE
        0,  # inX
        0,  # inY
        # wfr.mesh.eStart,
        # wfr.mesh.xStart,
        # wfr.mesh.yStart,
    )

    photon_energy_eV = np.linspace(wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne)
    img_x_mm = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.nx) * 1e3
    img_y_mm = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, wfr.mesh.ny) * 1e3
    intensity = np.array(intensity).reshape(ny, nx, nhv)
    return xr.DataArray(
        intensity,
        dims=["y", "x", "hv"],
        coords={"hv": photon_energy_eV, "x": img_x_mm, "y": img_y_mm},
    ).squeeze()


def calculate_spectrum_multi(
    field_container: SRWLMagFldC,
    file_path: str,
    *,
    obs_z_m: float = 10.0,  # Observation point z in meters
    theta_x_half_mrad: float = 0.9,  # Observation angle x in mrad
    theta_y_half_mrad: float = 0.9,  # Observation angle y in mrad
    hv_start_eV: float = 1.0,  # Start photon energy in eV
    hv_end_eV: float = 100.0,  # End photon energy in eV
    nhv: int = 125,
    nx: int = 101,
    ny: int = 101,
    n_electrons: int = 100,
    beam: str = "ALS",
    use_mpi: bool = False,  # Whether to use MPI for parallelization.
    only_flux: bool = False,  # Use SRW _char=10 instead of four Stokes components.
    opt_bl=None,  # Optional SRWLOptC beamline for propagation.
    extra_attrs=None,
):
    """Calculate a multi-electron trajectory through the given field."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be at least 1.")
    if opt_bl is not None and nx == 1 and ny == 1:
        raise ValueError(
            "SRW propagation through opt_bl requires nx > 1 and/or ny > 1. "
            "For a fast flux estimate, use only_flux=True with a 1D mesh: "
            "nx=1, ny>1 for vertical dispersion, or nx>1, ny=1 for horizontal "
            "dispersion."
        )

    if extra_attrs is None:
        extra_attrs = {}

    elec_beam_me = SRWLPartBeam()

    iavg, e, sig_e, sig_x, sig_x_pr, sig_y, sig_y_pr = get_beam_params(beam)
    elec_beam_me.from_RMS(
        _Iavg=iavg,  # Average current [A]
        _e=e,  # Electron energy [GeV]
        _sig_e=sig_e,  # RMS energy spread
        _sig_x=sig_x,  # Horizontal RMS size [m]
        _sig_x_pr=sig_x_pr,  # Horizontal RMS divergence [rad]
        _m_xx_pr=0.0,  # <(x-<x>)(x’-<x’>)>
        _sig_y=sig_y,  # Vertical RMS size [m]
        _sig_y_pr=sig_y_pr,  # Vertical RMS divergence [rad]
        _m_yy_pr=0.0,  # <(y-<y>)(y’-<y’>)>
    )

    # Get start Z assuming symmetric single field
    z_hw = field_container.arMagFld[0].rz / 2

    elec_beam_me.partStatMom1.x = 0.0
    elec_beam_me.partStatMom1.y = 0.0

    elec_beam_me.partStatMom1.z = -z_hw

    elec_beam_me.partStatMom1.xp = 0.0
    elec_beam_me.partStatMom1.yp = 0.0

    elec_beam_me.partStatMom1.relE0 = 1
    elec_beam_me.partStatMom1.nq = -1

    # Define mesh
    x_half_m = obs_z_m * theta_x_half_mrad * 1e-3
    y_half_m = obs_z_m * theta_y_half_mrad * 1e-3
    x_start_m = -x_half_m if nx > 1 or opt_bl is None else 0.0
    x_fin_m = x_half_m if nx > 1 or opt_bl is None else 0.0
    y_start_m = -y_half_m if ny > 1 or opt_bl is None else 0.0
    y_fin_m = y_half_m if ny > 1 or opt_bl is None else 0.0
    mesh_stack = SRWLRadMesh(
        hv_start_eV,
        hv_end_eV,
        nhv,
        x_start_m,
        x_fin_m,
        nx,
        y_start_m,
        y_fin_m,
        ny,
        obs_z_m,
    )

    sr_rel_prec = [
        0.001,  # relative precision
        -z_hw,  # integration start [m]
        z_hw,  # integration end [m]
        10000,  # trajectory points
        0,  # terminating terms off
    ]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_processes = comm.Get_size() if use_mpi else 1
    if n_electrons < 1:
        raise ValueError("n_electrons must be at least 1.")
    if mpi_processes > 1:
        # SRW reserves rank 0 as the master and distributes electrons over the
        # remaining ranks. Match SRW's own round() rule to avoid extra sends.
        srw_worker_processes = mpi_processes - 1
        n_part_avg_proc = int(round(n_electrons / srw_worker_processes))
        if n_part_avg_proc < 1:
            raise ValueError(
                "n_electrons is too small for the requested MPI process count; "
                "use fewer MPI processes."
            )
        n_save_per = srw_worker_processes + 1
        effective_n_electrons = n_part_avg_proc * srw_worker_processes
    else:
        srw_worker_processes = 0
        n_part_avg_proc = 1
        n_save_per = n_electrons + 1
        effective_n_electrons = n_electrons
    srw_char = 0 if only_flux and opt_bl is not None else 10 if only_flux else 1

    srwl_wfr_emit_prop_multi_e(
        _e_beam=elec_beam_me,
        _mag=field_container,
        _mesh=mesh_stack,
        _sr_meth=0,  # manual SR integration
        _sr_rel_prec=sr_rel_prec,
        _n_part_tot=n_electrons,  # Total macro-electrons across all workers
        _n_part_avg_proc=n_part_avg_proc,
        _n_save_per=n_save_per,
        _file_path=file_path,
        _sr_samp_fact=-1,
        _opt_bl=opt_bl,
        _pres_ang=0,  # 0 = coordinate plane at OBSERVATION_Z_M
        _char=srw_char,  # 1 = all Stokes parameters, 10 = flux
        _x0=0.0,
        _y0=0.0,
        _e_ph_integ=0,
        _rand_meth=2,  # Halton sequence, usually smoother than random
        _tryToUseMPI=use_mpi,
        _me_approx=0,  # full Monte-Carlo over electron phase space
        _file_form="hdf5",
    )

    if rank == 0:
        info_path = pathlib.Path(file_path)
        info_path = info_path.with_name(info_path.stem + "_info.json")
        with info_path.open("w") as f:
            json.dump(
                extra_attrs
                | {
                    "obs_z_m": obs_z_m,
                    "theta_x_half_mrad": theta_x_half_mrad,
                    "theta_y_half_mrad": theta_y_half_mrad,
                    "x_range_m": 2.0 * x_half_m,
                    "y_range_m": 2.0 * y_half_m,
                    "only_flux": only_flux,
                    "srw_char": srw_char,
                    "srw_mpi_processes": mpi_processes,
                    "srw_worker_processes": srw_worker_processes,
                    "srw_n_part_avg_proc": n_part_avg_proc,
                    "srw_n_save_per": n_save_per,
                    "srw_effective_n_electrons": effective_n_electrons,
                    "has_opt_bl": opt_bl is not None,
                    "propagated_flux_integrated_on_readback": (
                        only_flux and opt_bl is not None
                    ),
                    "date": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
