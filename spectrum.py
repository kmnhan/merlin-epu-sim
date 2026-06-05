"""Single-electron trajectory calculation."""

import datetime
import json

from mpi4py import MPI
from srwpy.srwlib import (
    SRWLMagFldC,
    SRWLPartBeam,
    SRWLRadMesh,
    srwl_wfr_emit_prop_multi_e,
)


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
    """Calculate a single-electron trajectory through the given field."""
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
        _char=0,  # total intensity / S0
        _x0=0.0,
        _y0=0.0,
        _e_ph_integ=0,
        _rand_meth=2,  # Halton sequence, usually smoother than random
        _tryToUseMPI=use_mpi,
        _me_approx=0,  # full Monte-Carlo over electron phase space
        _file_form="hdf5",
    )

    if rank == 0:
        json.dump(
            extra_attrs
            | {
                "obs_z_m": obs_z_m,
                "theta_x_half_mrad": theta_x_half_mrad,
                "theta_y_half_mrad": theta_y_half_mrad,
                "date": datetime.datetime.now().isoformat(),
            },
            open(file_path.replace(".h5", "_info.json"), "w"),
            indent=2,
        )
