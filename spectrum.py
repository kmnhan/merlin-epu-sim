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
    srwl,
    srwl_wfr_emit_prop_multi_e,
)


def calculate_spectrum_multi_mpi(
    field_container: SRWLMagFldC,
    file_path: str | pathlib.Path,
    *,
    n_processes: int = 8,
    mpi_executable: str | pathlib.Path = "mpirun",
    overwrite: bool = False,
    return_xarray: bool = True,
    **spectrum_kwargs,
):
    """Run calculate_spectrum_multi through mpirun and block until it finishes.

    If return_xarray=True, the HDF5 output is read back with
    srwl_uti_read_intens_hdf5() and srw_hdf5_output_to_xarray().
    """
    if n_processes < 1:
        raise ValueError("n_processes must be at least 1.")

    if MPI.COMM_WORLD.Get_size() != 1:
        raise RuntimeError(
            "calculate_spectrum_multi_mpi() should be called from a normal Python "
            "process, not from inside an existing MPI worker."
        )

    file_path = pathlib.Path(file_path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    info_path = file_path.with_name(file_path.stem + "_info.json")

    if file_path.exists() and not overwrite:
        raise FileExistsError(f"{file_path} already exists; pass overwrite=True.")

    if overwrite:
        file_path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)

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
            _xc=float(field_container.arXc[0]) if len(field_container.arXc) else 0.0,
            _yc=float(field_container.arYc[0]) if len(field_container.arYc) else 0.0,
            _zc=float(field_container.arZc[0]) if len(field_container.arZc) else 0.0,
        )

        spectrum_kwargs = dict(spectrum_kwargs)
        spectrum_kwargs.pop("field_container", None)
        spectrum_kwargs.pop("file_path", None)
        spectrum_kwargs.pop("use_mpi", None)

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
                "from spectrum import calculate_spectrum_multi",
                "with open(sys.argv[1], 'r') as f:",
                "    config = json.load(f)",
                "field_container = srwl_uti_read_mag_fld_3d(config['field_path'])",
                "kwargs = dict(config.get('spectrum_kwargs', {}))",
                "kwargs.pop('use_mpi', None)",
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

    if not file_path.exists():
        raise FileNotFoundError(f"MPI spectrum calculation did not create {file_path}")

    if not return_xarray:
        return file_path

    from srwpy.srwlib import srwl_uti_read_intens_hdf5
    from xarray_io import srw_hdf5_output_to_xarray

    ar_int, mesh, meta, _ = srwl_uti_read_intens_hdf5(str(file_path))
    data = srw_hdf5_output_to_xarray(ar_int, mesh, meta).rename(file_path.stem)
    if info_path.is_file():
        with info_path.open("r") as f:
            data = data.assign_attrs(**json.load(f))
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
    iavg=0.5,  # Ring current
    e=1.9,  # Electron energy in GeV
):
    """Calculate a single-electron trajectory through the given field."""

    # Get start Z assuming symmetric single field
    z_hw = field_container.arMagFld[0].rz / 2
    if z0_m is None:
        # Assuming symmetric field around z=0, start at the beginning of the field
        z0_m = -z_hw

    # Setup single-electron beam
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
    iavg=0.5,  # Ring current
    e=1.9,  # Electron energy in GeV
    sig_e=0.0,  # RMS energy spread
    sig_x=0.31e-3,  # Horizontal RMS size in m
    sig_x_pr=0.023e-3,  # Horizontal RMS divergence in rad
    sig_y=0.023e-3,  # Vertical RMS size in m
    sig_y_pr=0.0065e-3,  # Vertical RMS divergence in rad
    use_mpi: bool = False,  # Whether to use MPI for parallelization.
    extra_attrs=None,
):
    """Calculate a multi-electron trajectory through the given field."""
    if extra_attrs is None:
        extra_attrs = {}

    elec_beam_me = SRWLPartBeam()

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
    mesh_stack = SRWLRadMesh(
        hv_start_eV,
        hv_end_eV,
        nhv,
        -x_half_m,
        x_half_m,
        nx,
        -y_half_m,
        y_half_m,
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

    srwl_wfr_emit_prop_multi_e(
        _e_beam=elec_beam_me,
        _mag=field_container,
        _mesh=mesh_stack,
        _sr_meth=0,  # manual SR integration
        _sr_rel_prec=sr_rel_prec,
        _n_part_tot=n_electrons,  # Total macro-electrons across all workers
        _n_part_avg_proc=5,  # How many macro-electrons each worker averages before sending partial Stokes data back to rank 0
        _n_save_per=20,
        _file_path=file_path,
        _sr_samp_fact=-1,
        _opt_bl=None,
        _pres_ang=0,  # 0 = coordinate plane at OBSERVATION_Z_M
        _char=1,  # Get all stokes parameters, set to 0 for intensity only
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
                    "date": datetime.datetime.now().isoformat(),
                },
                f,
                indent=2,
            )
