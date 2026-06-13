import numpy as np
import xarray as xr

HC_EV_M = 1.239841984e-6  # h*c in eV*m


def integrate_aperture_if_needed(
    da,
    *,
    obs_z_m=10.0,
    aperture_half_x_mrad=None,
    aperture_half_y_mrad=None,
    stokes=0,
):
    """Return flux with x/y integrated. x/y coordinates are assumed to be in mm."""
    out = da

    if "stokes" in out.dims:
        out = out.sel(stokes=stokes)

    if "x" in out.dims:
        out = out.sortby("x")
        if aperture_half_x_mrad is not None:
            half_x_mm = obs_z_m * aperture_half_x_mrad
            out = out.sel(x=slice(-half_x_mm, half_x_mm))
        out = out.integrate("x")

    if "y" in out.dims:
        out = out.sortby("y")
        if aperture_half_y_mrad is not None:
            half_y_mm = obs_z_m * aperture_half_y_mrad
            out = out.sel(y=slice(-half_y_mm, half_y_mm))
        out = out.integrate("y")

    return out.sortby("hv")


def _curve_on_grid(value, hv):
    """Scalar, callable, 2-column array, or DataArray -> numpy array on hv grid."""
    hv = np.asarray(hv, dtype=float)

    if isinstance(value, xr.DataArray):
        return value.sortby("hv").interp(hv=hv).values.astype(float)

    if callable(value):
        out = np.asarray(value(hv), dtype=float)
        if out.ndim == 0:
            return np.full_like(hv, float(out), dtype=float)
        return out

    arr = np.asarray(value)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return np.interp(hv, arr[:, 0], arr[:, 1])

    return np.full_like(hv, float(value), dtype=float)


def _curve_da_on_grid(value, hv):
    """Scalar, callable, 2-column array, or DataArray -> DataArray on hv grid."""
    hv = np.asarray(hv, dtype=float)

    if isinstance(value, xr.DataArray):
        return value.sortby("hv").interp(hv=hv)

    return xr.DataArray(_curve_on_grid(value, hv), dims="hv", coords={"hv": hv})


def _coords_without_hv(da, dims):
    coords = {}
    dim_set = set(dims)
    for name, coord in da.coords.items():
        if name == "hv" or "hv" in coord.dims:
            continue
        if set(coord.dims).issubset(dim_set):
            coords[name] = coord
    return coords


def vlspgm_angles(hv_eV, *, g0, cff0, order=1):
    """SRW-compatible PGM angle convention. g0 is in lines/mm."""
    hv = np.asarray(hv_eV, dtype=float)
    if np.isclose(cff0, 1.0):
        raise ValueError("cff0 close to 1 needs a separate numerical solve.")

    u = order * (HC_EV_M / hv) * g0 * 1000.0
    c2 = cff0**2
    den = c2 - 1.0

    sin_beta = (u * c2 - np.sqrt(u * u * c2 + den * den)) / den
    sin_alpha = u - sin_beta

    valid = (np.abs(sin_alpha) <= 1.0) & (np.abs(sin_beta) <= 1.0)

    alpha = np.full_like(hv, np.nan)
    beta = np.full_like(hv, np.nan)
    alpha[valid] = np.arcsin(sin_alpha[valid])  # from grating normal
    beta[valid] = np.arcsin(sin_beta[valid])  # from grating normal

    grazing = 0.5 * np.pi - alpha
    return alpha, beta, grazing


def vlspgm_slit_fwhm_eV(
    hv,
    *,
    g0,
    r_m=None,
    r_prime_m,
    cff0,
    exit_slit_width_um,
    source_width_um=0.0,
    entrance_slit_width_um=0.0,
    source_image_width_um=0.0,
    order=1,
    extra_fwhm_eV=0.0,
):
    """Slit-limited monochromator energy FWHM.

    source_width_um and entrance_slit_width_um are object-plane widths and use
    r_m. source_image_width_um is an already-imaged width at the exit-slit plane
    and uses r_prime_m.
    """
    hv_vals = hv.values if isinstance(hv, xr.DataArray) else np.asarray(hv, dtype=float)
    alpha, beta, grazing = vlspgm_angles(hv_vals, g0=g0, cff0=cff0, order=order)
    denom = np.abs(np.sin(alpha) + np.sin(beta))

    exit_width_m = 1e-6 * np.hypot(exit_slit_width_um, source_image_width_um)
    exit_fwhm = hv_vals * exit_width_m * np.abs(np.cos(beta)) / (r_prime_m * denom)

    object_width_m = 1e-6 * np.hypot(source_width_um, entrance_slit_width_um)
    if object_width_m > 0:
        if r_m is None:
            raise ValueError(
                "r_m is required when source_width_um or entrance_slit_width_um is nonzero."
            )
        object_fwhm = hv_vals * object_width_m * np.abs(np.cos(alpha)) / (r_m * denom)
    else:
        object_fwhm = 0.0

    fwhm = np.hypot(object_fwhm, exit_fwhm)
    fwhm = np.hypot(fwhm, _curve_on_grid(extra_fwhm_eV, hv_vals))

    return xr.DataArray(
        fwhm,
        dims="hv",
        coords={"hv": hv_vals},
        attrs={
            "units": "eV",
            "grazing_angle_rad": grazing,
            "source_width_um": source_width_um,
            "entrance_slit_width_um": entrance_slit_width_um,
            "exit_slit_width_um": exit_slit_width_um,
            "source_image_width_um": source_image_width_um,
            "source_image_width_meaning": "already-imaged width at exit slit plane",
        },
    )


def vlspgm_spectrum(
    flux,
    *,
    throughput=1.0,
    fwhm_eV,
    hv_out=None,
    input_unit="ph/s/0.1%bw",
    output="passband_flux",
    samples_per_fwhm=8,
    truncate_sigma=5.0,
):
    """Fast local-window convolution along hv.

    flux can have arbitrary extra dimensions; only hv is convolved.
    """
    if "hv" not in flux.dims:
        raise ValueError("flux must have an hv dimension.")

    flux = flux.sortby("hv")
    hv = flux.hv.values.astype(float)

    if input_unit == "ph/s/0.1%bw":
        density = flux / (1e-3 * flux.hv)  # ph/s/eV
    elif input_unit == "ph/s/eV":
        density = flux
    else:
        raise ValueError("input_unit must be 'ph/s/0.1%bw' or 'ph/s/eV'.")

    hv_out = hv if hv_out is None else np.asarray(hv_out, dtype=float)
    fwhm = np.asarray(_curve_on_grid(fwhm_eV, hv_out), dtype=float)
    if fwhm.shape != hv_out.shape:
        raise ValueError("fwhm_eV must broadcast to a 1D array on hv_out.")

    finite = fwhm[np.isfinite(fwhm) & (fwhm > 0)]
    if finite.size == 0:
        raise ValueError("No valid positive FWHM values.")

    step0 = np.median(np.diff(hv))
    step = min(step0, np.nanmin(finite) / samples_per_fwhm)
    hv_calc = np.arange(hv[0], hv[-1] + 0.5 * step, step)

    src_da = density.interp(hv=hv_calc) * _curve_da_on_grid(throughput, hv_calc)
    src_da = src_da.transpose("hv", ...)
    src_dims = src_da.dims
    rest_dims = src_dims[1:]
    rest_shape = src_da.shape[1:]
    src = src_da.values.astype(float, copy=False).reshape(src_da.sizes["hv"], -1)

    out = np.full((hv_out.size, src.shape[1]), np.nan, dtype=float)

    for i, (e0, w) in enumerate(zip(hv_out, fwhm)):
        if not np.isfinite(w) or w <= 0:
            continue

        sigma = w / 2.354820045
        lo = np.searchsorted(hv_calc, e0 - truncate_sigma * sigma, side="left")
        hi = np.searchsorted(hv_calc, e0 + truncate_sigma * sigma, side="right")

        if hi <= lo + 1:
            continue

        x = hv_calc[lo:hi]
        lsf = np.exp(-0.5 * ((x - e0) / sigma) ** 2)

        if output == "density":
            norm = np.trapezoid(lsf, x)
            out[i] = np.trapezoid(src[lo:hi] * lsf[:, None], x, axis=0) / norm
        elif output == "passband_flux":
            out[i] = np.trapezoid(src[lo:hi] * lsf[:, None], x, axis=0)
        else:
            raise ValueError("output must be 'passband_flux' or 'density'.")

    units = "ph/s" if output == "passband_flux" else "ph/s/eV"
    out = out.reshape((hv_out.size, *rest_shape))
    coords = {"hv": hv_out} | _coords_without_hv(src_da, rest_dims)
    result = xr.DataArray(
        out, dims=("hv", *rest_dims), coords=coords, attrs={"units": units}
    )
    ordered_dims = [dim for dim in flux.dims if dim in result.dims]
    extra_dims = [dim for dim in result.dims if dim not in ordered_dims]
    return result.transpose(*ordered_dims, *extra_dims)


def convert_flux_through_vlspgm(
    flux,
    *,
    g0,
    g1,
    g2,
    r_m,
    r_prime_m,
    cff0,
    exit_slit_width_um,
    throughput=1.0,
    source_width_um=0.0,
    entrance_slit_width_um=0.0,
    source_image_width_um=0.0,
    order=1,
    hv_out=None,
    input_unit="ph/s/0.1%bw",
    output="passband_flux",
    samples_per_fwhm=8,
    truncate_sigma=5.0,
):
    """Single diffraction-order spectrum on the actual photon-energy axis.

    For order > 1, hv_out is interpreted as actual photon energy, not as the
    first-order monochromator setpoint. Use convert_flux_through_vlspgm_orders()
    for higher-order contamination on a first-order scan axis.
    """
    fwhm = vlspgm_slit_fwhm_eV(
        flux.hv if hv_out is None else hv_out,
        g0=g0,
        r_m=r_m,
        r_prime_m=r_prime_m,
        cff0=cff0,
        exit_slit_width_um=exit_slit_width_um,
        source_width_um=source_width_um,
        entrance_slit_width_um=entrance_slit_width_um,
        source_image_width_um=source_image_width_um,
        order=order,
    )

    mono = vlspgm_spectrum(
        flux,
        throughput=throughput,
        fwhm_eV=fwhm,
        hv_out=hv_out,
        input_unit=input_unit,
        output=output,
        samples_per_fwhm=samples_per_fwhm,
        truncate_sigma=truncate_sigma,
    )

    mono.attrs.update(
        {
            "g0_lines_per_mm": g0,
            "g1_lines_per_mm2": g1,
            "g2_lines_per_mm3": g2,
            "r_m": r_m,
            "r_prime_m": r_prime_m,
            "cff0": cff0,
            "source_width_um": source_width_um,
            "entrance_slit_width_um": entrance_slit_width_um,
            "exit_slit_width_um": exit_slit_width_um,
            "source_image_width_um": source_image_width_um,
            "diffraction_order": order,
            "hv_axis": "actual_photon_energy_eV",
            "note": (
                "Scalar post-process: object-plane width uses r_m, exit-plane "
                "width uses r_prime_m. g1/g2 focusing/aberrations are not "
                "propagated."
            ),
        }
    )
    return mono, fwhm


def convert_flux_through_vlspgm_order_on_setpoint_axis(
    flux,
    *,
    g0,
    g1,
    g2,
    r_m,
    r_prime_m,
    cff0,
    exit_slit_width_um,
    throughput=1.0,
    source_width_um=0.0,
    entrance_slit_width_um=0.0,
    source_image_width_um=0.0,
    order=1,
    hv_set=None,
    input_unit="ph/s/0.1%bw",
    output="passband_flux",
    samples_per_fwhm=8,
    truncate_sigma=5.0,
):
    """Contribution of one diffraction order on a first-order set-energy axis.

    At first-order setpoint E, order n transmits source photons centered near
    n*E. The returned DataArray uses hv = E and stores n*E as actual_hv_center.
    """
    if order < 1:
        raise ValueError("order must be a positive integer for setpoint-axis scans.")

    if "hv" not in flux.dims:
        raise ValueError("flux must have an hv dimension.")
    flux = flux.sortby("hv")

    hv_set_vals = (
        flux.hv.values.astype(float)
        if hv_set is None
        else np.asarray(hv_set, dtype=float)
    )
    actual_hv = order * hv_set_vals

    fwhm_actual = vlspgm_slit_fwhm_eV(
        actual_hv,
        g0=g0,
        r_m=r_m,
        r_prime_m=r_prime_m,
        cff0=cff0,
        exit_slit_width_um=exit_slit_width_um,
        source_width_um=source_width_um,
        entrance_slit_width_um=entrance_slit_width_um,
        source_image_width_um=source_image_width_um,
        order=order,
    )

    mono_actual = vlspgm_spectrum(
        flux,
        throughput=throughput,
        fwhm_eV=fwhm_actual,
        hv_out=actual_hv,
        input_unit=input_unit,
        output=output,
        samples_per_fwhm=samples_per_fwhm,
        truncate_sigma=truncate_sigma,
    )

    mono = mono_actual.assign_coords(hv=hv_set_vals)
    mono = mono.assign_coords(actual_hv_center=("hv", actual_hv))
    mono.attrs.update(
        {
            "g0_lines_per_mm": g0,
            "g1_lines_per_mm2": g1,
            "g2_lines_per_mm3": g2,
            "r_m": r_m,
            "r_prime_m": r_prime_m,
            "cff0": cff0,
            "source_width_um": source_width_um,
            "entrance_slit_width_um": entrance_slit_width_um,
            "exit_slit_width_um": exit_slit_width_um,
            "source_image_width_um": source_image_width_um,
            "diffraction_order": order,
            "hv_axis": "first_order_setpoint_eV",
            "actual_hv_center": "order * hv",
            "note": (
                "Order contribution for a first-order scan axis. This samples "
                "the source near order*hv, not hv."
            ),
        }
    )

    fwhm = fwhm_actual.assign_coords(hv=hv_set_vals)
    fwhm = fwhm.assign_coords(actual_hv_center=("hv", actual_hv))
    fwhm.attrs.update(
        {
            "units": "eV",
            "diffraction_order": order,
            "hv_axis": "first_order_setpoint_eV",
            "actual_hv_center": "order * hv",
        }
    )

    return mono, fwhm


def convert_flux_through_vlspgm_orders(
    flux,
    *,
    g0,
    g1,
    g2,
    r_m,
    r_prime_m,
    cff0,
    exit_slit_width_um,
    order_throughputs=None,
    throughput=1.0,
    source_width_um=0.0,
    entrance_slit_width_um=0.0,
    source_image_width_um=0.0,
    hv_set=None,
    input_unit="ph/s/0.1%bw",
    output="passband_flux",
    samples_per_fwhm=8,
    truncate_sigma=5.0,
):
    """Sum diffraction-order contributions on a first-order scan axis.

    Parameters
    ----------
    order_throughputs
        Mapping like {1: T1, 2: T2, 3: T3}. Each T can be a scalar, callable,
        two-column array, or DataArray over actual photon energy hv.
        If omitted, only first order is included using throughput.
    """
    if order_throughputs is None:
        order_throughputs = {1: throughput}

    orders = sorted(int(order) for order in order_throughputs)
    parts = []
    fwhms = []

    for order in orders:
        part, fwhm = convert_flux_through_vlspgm_order_on_setpoint_axis(
            flux,
            g0=g0,
            g1=g1,
            g2=g2,
            r_m=r_m,
            r_prime_m=r_prime_m,
            cff0=cff0,
            exit_slit_width_um=exit_slit_width_um,
            throughput=order_throughputs[order],
            source_width_um=source_width_um,
            entrance_slit_width_um=entrance_slit_width_um,
            source_image_width_um=source_image_width_um,
            order=order,
            hv_set=hv_set,
            input_unit=input_unit,
            output=output,
            samples_per_fwhm=samples_per_fwhm,
            truncate_sigma=truncate_sigma,
        )
        parts.append(part)
        fwhms.append(fwhm)

    order_coord = xr.IndexVariable("order", orders)
    parts_by_order = xr.concat(parts, dim=order_coord)
    fwhm_by_order = xr.concat(fwhms, dim=order_coord)
    total = parts_by_order.fillna(0.0).sum("order")
    total.attrs.update(parts_by_order.attrs)
    total.attrs.update(
        {
            "included_orders": tuple(orders),
            "hv_axis": "first_order_setpoint_eV",
            "note": (
                "Sum of diffraction-order contributions. Order n samples "
                "incident photons near n*hv."
            ),
        }
    )

    return total, parts_by_order, fwhm_by_order


def mono_energy_spectrum(
    flux,
    *,
    set_energy_eV,
    g0,
    g1,
    g2,
    r_m,
    r_prime_m,
    cff0,
    exit_slit_width_um,
    order_throughputs=None,
    source_width_um=0.0,
    entrance_slit_width_um=0.0,
    source_image_width_um=0.0,
    input_unit="ph/s/0.1%bw",
    truncate_sigma=5.0,
    points_per_fwhm=20,
    return_density_spectrum=True,
):
    """Flux through one fixed monochromator setting.

    Returns ph/s integrated over the monochromator line shape. If
    return_density_spectrum=True, also return the ph/s/eV spectrum whose
    integral gives the ph/s result.
    """
    if "hv" not in flux.dims:
        raise ValueError("flux must have an hv dimension.")
    if order_throughputs is None:
        order_throughputs = {1: 1.0}

    flux = flux.sortby("hv")

    if input_unit == "ph/s/0.1%bw":
        density = flux / (1e-3 * flux.hv)  # ph/s/eV
    elif input_unit == "ph/s/eV":
        density = flux
    else:
        raise ValueError("input_unit must be 'ph/s/0.1%bw' or 'ph/s/eV'.")

    flux_parts = []
    density_parts = []

    for order, throughput in sorted(order_throughputs.items()):
        order = int(order)
        center = order * set_energy_eV

        fwhm = vlspgm_slit_fwhm_eV(
            [center],
            g0=g0,
            r_m=r_m,
            r_prime_m=r_prime_m,
            cff0=cff0,
            exit_slit_width_um=exit_slit_width_um,
            source_width_um=source_width_um,
            entrance_slit_width_um=entrance_slit_width_um,
            source_image_width_um=source_image_width_um,
            order=order,
        ).item()
        if not np.isfinite(fwhm) or fwhm <= 0:
            raise ValueError(f"Invalid FWHM for order {order}: {fwhm!r}.")

        sigma = fwhm / 2.354820045
        step = fwhm / points_per_fwhm
        hv = np.arange(
            center - truncate_sigma * sigma,
            center + truncate_sigma * sigma + 0.5 * step,
            step,
        )

        src = density.interp(hv=hv)
        transmission = _curve_da_on_grid(throughput, hv)
        lsf = xr.DataArray(
            np.exp(-0.5 * ((hv - center) / sigma) ** 2),
            dims="hv",
            coords={"hv": hv},
        )

        part_density = src * transmission * lsf
        part_density = part_density.assign_coords(order=order)
        part_density.attrs.update(
            {
                "units": "ph/s/eV",
                "set_energy_eV": set_energy_eV,
                "actual_center_eV": center,
                "fwhm_eV": fwhm,
                "diffraction_order": order,
                "hv_axis": "actual_photon_energy_eV",
            }
        )

        part_flux = part_density.integrate("hv")
        part_flux = part_flux.assign_coords(order=order)
        part_flux.attrs.update(
            {
                "units": "ph/s",
                "set_energy_eV": set_energy_eV,
                "actual_center_eV": center,
                "fwhm_eV": fwhm,
                "diffraction_order": order,
                "note": "Integrated over actual photon energy hv.",
            }
        )

        flux_parts.append(part_flux)
        density_parts.append(part_density)

    order_coord = xr.IndexVariable(
        "order", [int(order) for order in sorted(order_throughputs)]
    )
    flux_by_order = xr.concat(flux_parts, dim=order_coord)
    total_flux = flux_by_order.fillna(0.0).sum("order")
    total_flux.attrs.update(
        {
            "units": "ph/s",
            "set_energy_eV": set_energy_eV,
            "g0_lines_per_mm": g0,
            "g1_lines_per_mm2": g1,
            "g2_lines_per_mm3": g2,
            "r_m": r_m,
            "r_prime_m": r_prime_m,
            "cff0": cff0,
            "included_orders": tuple(order_coord.values.tolist()),
            "note": "Total fixed-setting passband flux integrated over hv.",
        }
    )

    if not return_density_spectrum:
        return total_flux, flux_by_order

    density_by_order = xr.concat(density_parts, dim=order_coord, join="outer")
    total_density = density_by_order.fillna(0.0).sum("order")
    total_density.attrs.update(
        {
            "units": "ph/s/eV",
            "set_energy_eV": set_energy_eV,
            "note": "Energy density spectrum; integrate over hv to recover ph/s.",
        }
    )
    return total_flux, flux_by_order, total_density, density_by_order
