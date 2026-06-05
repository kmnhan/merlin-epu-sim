import xarray as xr
from srwpy.srwlib import SRWLMagFldC, SRWLParticle, SRWLPrtTrj, srwl
import numpy as np

def calculate_trajectory(
    field_container: SRWLMagFldC,
    n_traj_points: int = 20001,
    *,
    x0_m: float = 0.0,  # Initial x position in meters
    y0_m: float = 0.0,  # Initial y position in meters
    z0_m: float | None = None,  # Initial z position, defaults to the start of the field
    xp0_rad: float = 0.0,  # Instant transverse velocity x'/c (angle for relativistic particles)
    yp0_rad: float = 0.0,  # Instant transverse velocity y'/c (angle for relativistic particles)
    e: float = 1.9,  # Electron energy in GeV
):
    """Calculate a single-electron trajectory through the given field."""
    # Get Z range
    z_width_m = field_container.arMagFld[0].rz

    if z0_m is None:
        # Assuming symmetric field around z=0, start at the beginning of the field
        z0_m = -z_width_m / 2

    # Define relativistic particle
    part = SRWLParticle(
        _x=x0_m,
        _y=y0_m,
        _z=z0_m,
        _xp=xp0_rad,
        _yp=yp0_rad,
        _gamma=e / 0.510998950e-3,  # relative energy (E/mc^2)
        _relE0=1,  # rest mass
        _nq=-1,  # charge
    )

    # Calculate trajectory
    traj = SRWLPrtTrj()
    traj.partInitCond = part
    traj.allocate(n_traj_points, True)
    traj.ctStart = 0.0
    traj.ctEnd = z_width_m

    traj = srwl.CalcPartTraj(traj, field_container, [1])

    return xr.Dataset(
        data_vars=dict(
            x=("point", 1e3 * np.asarray(traj.arX)),
            y=("point", 1e3 * np.asarray(traj.arY)),
            z=("point", 1e3 * np.asarray(traj.arZ)),
            xp=("point", 1e3 * np.asarray(traj.arXp)),
            yp=("point", 1e3 * np.asarray(traj.arYp)),
            bx=("point", np.asarray(traj.arBx)),
            by=("point", np.asarray(traj.arBy)),
            bz=("point", np.asarray(traj.arBz)),
        ),
        attrs={
            "x0": 1e3 * traj.partInitCond.x,
            "y0": 1e3 * traj.partInitCond.y,
            "z0": 1e3 * traj.partInitCond.z,
            "xp0": traj.partInitCond.xp,
            "yp0": traj.partInitCond.yp,
            "gamma": traj.partInitCond.gamma,
            "relE0": traj.partInitCond.relE0,
            "nq": traj.partInitCond.nq,
        },
    )
