from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr
import xraydb
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from xraydb.materials import get_material
from xraydb.xray import PLANCK_HC

HC_EV_NM = 1239.841984
OPTICAL_CONSTANTS_DIR = Path(__file__).resolve().parent / "optical_constants"
TABULATED_NK_SOURCES = {
    "C": {
        "path": OPTICAL_CONSTANTS_DIR / "C_Hagemann_nk.csv",
        "max_energy_ev": None,
    },
    "Si": {
        "path": OPTICAL_CONSTANTS_DIR / "Si_Franta_20C_nk.csv",
        "max_energy_ev": 40.0,
    },
}


@lru_cache(maxsize=64)
def _material_formula_density(formula):
    formula_aliases = {
        "carbon": "C",
        "graphite": "C",
        "silicon": "Si",
    }
    formula = formula_aliases.get(str(formula).lower(), formula)
    material = get_material(formula)
    if material is None:
        return formula, None
    return material


@lru_cache(maxsize=2)
def _tabulated_nk_table(formula):
    source = TABULATED_NK_SOURCES[formula]
    table = np.loadtxt(source["path"], delimiter=",", comments="#")
    return table[:, 0], table[:, 1], table[:, 2], table[:, 3]


@lru_cache(maxsize=2)
def _tabulated_nk_interpolators(formula):
    _, wavelength, n, k = _tabulated_nk_table(formula)
    order = np.argsort(wavelength)
    log_wavelength = np.log(wavelength[order])
    return (
        PchipInterpolator(log_wavelength, n[order], extrapolate=False),
        PchipInterpolator(log_wavelength, k[order], extrapolate=False),
    )


def _tabulated_nk_domain(formula):
    energy, _, _, _ = _tabulated_nk_table(formula)
    lower = energy[0]
    upper = energy[-1]
    max_energy_ev = TABULATED_NK_SOURCES[formula]["max_energy_ev"]
    if max_energy_ev is not None:
        upper = max_energy_ev
    return lower, upper


def _tabulated_nk(formula, energy):
    table_energy, table_wavelength, _, _ = _tabulated_nk_table(formula)
    energy = np.clip(energy, table_energy[0], table_energy[-1])
    log_wavelength = np.log(HC_EV_NM / 1000.0 / energy)
    log_wavelength = np.clip(
        log_wavelength,
        np.log(table_wavelength.min()),
        np.log(table_wavelength.max()),
    )
    n_interp, k_interp = _tabulated_nk_interpolators(formula)
    return n_interp(log_wavelength), k_interp(log_wavelength)


def _resolve_polarization(polarization, *, lh):
    pol = str(polarization).lower()
    if pol in {"s", "sigma"}:
        return "s"
    if pol in {"p", "pi"}:
        return "p"

    pol = str(polarization).upper()
    if pol == "LH":
        return lh
    if pol == "LV":
        return "p" if lh == "s" else "s"
    raise ValueError("polarization must be one of 'LH', 'LV', 's', or 'p'")


@lru_cache(maxsize=8192)
def _m303_lrg_incidence_angle_scalar(photon_energy_ev: float) -> float:
    """M303 LRG grazing incidence angle in degrees for one photon energy."""
    photon_energy_ev = float(photon_energy_ev)
    if not np.isfinite(photon_energy_ev) or photon_energy_ev <= 0.0:
        raise ValueError(
            f"Photon energy must be positive and finite, got {photon_energy_ev}"
        )

    # MERLIN upgrade PDF LRG parameters
    d0 = 400.0  # lines/mm
    d1 = 0.2  # lines/mm^2
    r = 16580.0  # mm
    rprime = 5001.0  # mm
    radius = 120000.0  # mm

    wavelength_mm = (HC_EV_NM / photon_energy_ev) * 1e-6
    grating_term = wavelength_mm * d0

    def beta_from_alpha(alpha_grazing_rad):
        # Grazing-angle form of lambda*D0 = cos(alpha) - cos(beta)
        value = np.cos(alpha_grazing_rad) - grating_term
        if value < -1.0 or value > 1.0:
            raise ValueError("No physical beta solution")
        return np.arccos(value)

    def focus_residual(alpha_grazing_rad):
        beta_grazing_rad = beta_from_alpha(alpha_grazing_rad)
        return (
            -wavelength_mm * d1
            + np.sin(alpha_grazing_rad) ** 2 / r
            + np.sin(beta_grazing_rad) ** 2 / rprime
            - (np.sin(alpha_grazing_rad) + np.sin(beta_grazing_rad)) / radius
        )

    lower = np.deg2rad(0.01)
    upper = np.deg2rad(40.0)
    try:
        f_lower = focus_residual(lower)
        f_upper = focus_residual(upper)
        if np.isfinite(f_lower) and np.isfinite(f_upper):
            if f_lower == 0.0 or f_lower * f_upper < 0.0:
                alpha = brentq(focus_residual, lower, upper)
                beta = beta_from_alpha(alpha)
                return float(np.rad2deg(0.5 * (alpha + beta)))
    except ValueError:
        pass

    grid = np.linspace(lower, upper, 2000)
    vals = []
    for alpha in grid:
        try:
            vals.append(focus_residual(alpha))
        except ValueError:
            vals.append(np.nan)
    vals = np.asarray(vals)

    for a0, a1, f0, f1 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.isfinite(f0) and np.isfinite(f1) and f0 * f1 < 0:
            alpha = brentq(focus_residual, a0, a1)
            beta = beta_from_alpha(alpha)
            return float(np.rad2deg(0.5 * (alpha + beta)))

    raise ValueError(f"No LRG M303 angle solution for {photon_energy_ev} eV")


def m301_reflectivity(hv, polarization="LH"):
    pol = _resolve_polarization(polarization, lh="p")
    return _fixed_angle_reflectivity(hv, 1.25, "C", pol, "m301_reflectivity")


def m302_reflectivity(hv, polarization="LH"):
    pol = _resolve_polarization(polarization, lh="p")
    return _fixed_angle_reflectivity(hv, 6.5, "C", pol, "m302_reflectivity")


def _m303_lrg_incidence_angle_array(photon_energy_ev):
    energy = np.asarray(photon_energy_ev, dtype=float)
    if energy.ndim == 0:
        return _m303_lrg_incidence_angle_scalar(float(energy))

    flat = energy.ravel()
    unique, inverse = np.unique(flat, return_inverse=True)
    angles = np.array([_m303_lrg_incidence_angle_scalar(float(e)) for e in unique])
    return angles[inverse].reshape(energy.shape)


def _mirror_reflectivity_pairwise(
    theta,
    energy,
    formula="C",
    polarization="s",
):
    theta = np.asarray(theta, dtype=float)
    energy = np.asarray(energy, dtype=float)
    theta, energy = np.broadcast_arrays(theta, energy)
    shape = energy.shape
    theta = theta.ravel()
    energy = energy.ravel()

    formula, density = _material_formula_density(formula)
    out = np.empty_like(energy, dtype=float)
    if formula in TABULATED_NK_SOURCES:
        table_min, table_max = _tabulated_nk_domain(formula)
        tabulated_mask = (energy >= table_min) & (energy <= table_max)
    else:
        tabulated_mask = np.zeros_like(energy, dtype=bool)

    if np.any(tabulated_mask):
        n_tabulated, k_tabulated = _tabulated_nk(formula, energy[tabulated_mask])
        refractive_index = n_tabulated + 1j * k_tabulated
        cos_incidence = np.sin(theta[tabulated_mask])
        sin_incidence = np.cos(theta[tabulated_mask])
        cos_transmitted = np.sqrt(1.0 - (sin_incidence / refractive_index) ** 2)
        r_s = (cos_incidence - refractive_index * cos_transmitted) / (
            cos_incidence + refractive_index * cos_transmitted
        )
        r_p = (refractive_index * cos_incidence - cos_transmitted) / (
            refractive_index * cos_incidence + cos_transmitted
        )
        r_amp = r_s if polarization == "s" else r_p
        out[tabulated_mask] = (r_amp * r_amp.conjugate()).real

    xraydb_mask = ~tabulated_mask
    if np.any(xraydb_mask):
        delta, beta, _ = xraydb.xray_delta_beta(
            formula, density, energy[xraydb_mask]
        )
        n = 1.0 - delta - 1j * beta

        qf = 2.0 * np.pi * energy[xraydb_mask] / PLANCK_HC
        kiz = qf * np.sin(theta[xraydb_mask])
        ktz = qf * np.sqrt(n**2 - np.cos(theta[xraydb_mask]) ** 2)

        if polarization == "p":
            ktz = ktz / n

        r_amp = (kiz - ktz) / (kiz + ktz)
        out[xraydb_mask] = (r_amp * r_amp.conjugate()).real

    return out.reshape(shape)


def _fixed_angle_reflectivity(hv, angle_deg, formula, polarization, name):
    angle_rad = np.deg2rad(angle_deg)

    if isinstance(hv, xr.DataArray):
        out = xr.apply_ufunc(
            _mirror_reflectivity_pairwise,
            angle_rad,
            hv,
            kwargs={"formula": formula, "polarization": polarization},
            dask="parallelized",
            output_dtypes=[float],
        )
        out.name = name
        out.attrs = {
            "long_name": name.replace("_", " "),
            "units": "1",
            "formula": formula,
            "polarization": polarization,
            "grazing_angle_deg": angle_deg,
        }
        return out

    result = _mirror_reflectivity_pairwise(
        angle_rad,
        np.asarray(hv, dtype=float),
        formula=formula,
        polarization=polarization,
    )
    if result.ndim == 0:
        return float(result)
    return result


def _m303_reflectivity_scalar(hv, order=1, formula="C", polarization="LH"):
    order = float(order)
    if not np.isfinite(order) or order <= 0.0:
        raise ValueError(f"Order must be positive and finite, got {order}")

    pol = _resolve_polarization(polarization, lh="s")
    hv = float(hv)
    angle = _m303_lrg_incidence_angle_scalar(hv / order)
    return float(
        _mirror_reflectivity_pairwise(
            np.deg2rad(angle),
            hv,
            formula=formula,
            polarization=pol,
        )
    )


def m303_reflectivity(hv, order=1, formula="C", polarization="LH"):
    """Return M303 reflectivity for scalar, NumPy-like, or DataArray input."""
    pol = _resolve_polarization(polarization, lh="s")

    if isinstance(hv, xr.DataArray) or isinstance(order, xr.DataArray):
        hv_in = hv if isinstance(hv, xr.DataArray) else xr.DataArray(hv)
        order_in = order if isinstance(order, xr.DataArray) else xr.DataArray(order)
        hv_da, order_da = xr.broadcast(hv_in, order_in)
        angle = xr.apply_ufunc(
            _m303_lrg_incidence_angle_array,
            hv_da / order_da,
            dask="parallelized",
            output_dtypes=[float],
        )
        out = xr.apply_ufunc(
            _mirror_reflectivity_pairwise,
            np.deg2rad(angle),
            hv_da,
            kwargs={
                "formula": formula,
                "polarization": pol,
            },
            dask="parallelized",
            output_dtypes=[float],
        )
        out.name = "m303_reflectivity"
        out.attrs = {
            "long_name": "M303 reflectivity",
            "units": "1",
            "formula": formula,
            "polarization": polarization,
        }
        return out

    hv_array, order_array = np.broadcast_arrays(
        np.asarray(hv, dtype=float), np.asarray(order, dtype=float)
    )
    if hv_array.ndim == 0:
        return _m303_reflectivity_scalar(
            float(hv_array), float(order_array), formula, polarization
        )

    angle = _m303_lrg_incidence_angle_array(hv_array / order_array)
    return _mirror_reflectivity_pairwise(
        np.deg2rad(angle),
        hv_array,
        formula=formula,
        polarization=pol,
    )


def upstream_mirror_reflectivity(hv, order=1, formula="C", polarization="LH"):
    """M301 * M302 * M303"""
    return (
        m301_reflectivity(hv, polarization)
        * m302_reflectivity(hv, polarization)
        * m303_reflectivity(hv, order, formula, polarization)
    )


def m303_lrg_incidence_angle(photon_energy_ev):
    """Return M303 LRG grazing incidence angle in degrees.

    Accepts a scalar, NumPy array-like object, or xarray.DataArray.
    """
    if isinstance(photon_energy_ev, xr.DataArray):
        out = xr.apply_ufunc(
            _m303_lrg_incidence_angle_array,
            photon_energy_ev,
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
        out.name = "m303_lrg_grazing_angle"
        out.attrs.update(
            {"long_name": "M303 LRG grazing incidence angle", "units": "deg"}
        )
        return out

    energy = np.asarray(photon_energy_ev, dtype=float)
    if energy.ndim == 0:
        return _m303_lrg_incidence_angle_scalar(float(energy))

    return _m303_lrg_incidence_angle_array(energy)
