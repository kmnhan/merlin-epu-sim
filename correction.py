"""Field integral corrections.

Adapted from SRW Igor Pro code.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


def field_integrals(z_m, b_t):
    """
    Compute first and second field integrals.

    Parameters
    ----------
    z_m : array_like
        Longitudinal coordinate [m].
    b_t : array_like
        Magnetic field component [T].

    Returns
    -------
    i1 : float
        First field integral, ∫ B dz [T m].
    i2 : float
        Second field integral, ∫∫ B dz dz [T m^2].

    Notes
    -----
    This matches the Igor logic:

        integrate/T B        -> first integral wave
        integrate/T int(B)   -> second integral wave
    """
    z_m = np.asarray(z_m, dtype=float)
    b_t = np.asarray(b_t, dtype=float)

    int_b = cumulative_trapezoid(b_t, z_m, initial=0.0)

    i1 = int_b[-1]
    i2 = trapezoid(int_b, z_m)

    return float(i1), float(i2)


def normalized_gaussian(z_m, center_m, rms_len_m):
    """
    Gaussian correction shape normalized so that ∫ g dz = 1.

    Returns
    -------
    g : ndarray
        Gaussian shape with units [1/m].
    """
    z_m = np.asarray(z_m, dtype=float)

    g = np.exp(-0.5 * ((z_m - center_m) / rms_len_m) ** 2)
    area = trapezoid(g, z_m)

    if area == 0.0:
        raise ValueError("Gaussian correction has zero sampled area.")

    return g / area


def kick_strengths(
    z_m,
    b_t,
    dist_between_kicks_m=1.5,
    kick_center_m=None,
):
    """
    Reproduce the kick-strength logic of SRW Igor's SrwUndCorrectFieldInt.

    Returns
    -------
    kick_entry_tm, kick_exit_tm : float
        Integrated kick strengths [T m].
    """
    z_m = np.asarray(z_m, dtype=float)
    b_t = np.asarray(b_t, dtype=float)

    if kick_center_m is None:
        kick_center_m = 0.5 * (z_m[0] + z_m[-1])

    i1, i2 = field_integrals(z_m, b_t)

    z_end = z_m[-1]
    half_dist = 0.5 * dist_between_kicks_m

    z_entry = kick_center_m - half_dist
    z_exit = kick_center_m + half_dist

    # General two-thin-kick solution:
    #
    # I1 + K_entry + K_exit = 0
    #
    # I2 + K_entry * (z_end - z_entry)
    #    + K_exit  * (z_end - z_exit) = 0
    #
    # This reduces to the exact Igor formula when kick_center_m
    # is the center of the field map.
    a_entry = z_end - z_entry
    a_exit = z_end - z_exit

    kick_entry_tm = (i1 * a_exit - i2) / dist_between_kicks_m
    kick_exit_tm = (i2 - i1 * a_entry) / dist_between_kicks_m

    return kick_entry_tm, kick_exit_tm


def igor_correct_one_field_component(
    z_m,
    b_t,
    dist_between_kicks_m=1.5,
    rms_len_kick_mm=100.0,
    kick_center_m=None,
):
    """
    Igor-style field-integral correction.

    This uses the old SRW/Igor thin-kick formula for kick strengths,
    then adds finite Gaussian field bumps.
    """
    z_m = np.asarray(z_m, dtype=float)
    b_t = np.asarray(b_t, dtype=float)

    if kick_center_m is None:
        kick_center_m = 0.5 * (z_m[0] + z_m[-1])

    rms_len_kick_m = 1e-3 * rms_len_kick_mm

    z_entry = kick_center_m - 0.5 * dist_between_kicks_m
    z_exit = kick_center_m + 0.5 * dist_between_kicks_m

    kick_entry_tm, kick_exit_tm = kick_strengths(
        z_m,
        b_t,
        dist_between_kicks_m=dist_between_kicks_m,
        kick_center_m=kick_center_m,
    )

    g_entry = normalized_gaussian(z_m, z_entry, rms_len_kick_m)
    g_exit = normalized_gaussian(z_m, z_exit, rms_len_kick_m)

    b_corr_t = b_t + kick_entry_tm * g_entry + kick_exit_tm * g_exit

    i1_before, i2_before = field_integrals(z_m, b_t)
    i1_after, i2_after = field_integrals(z_m, b_corr_t)

    info = {
        "i1_before_T_m": i1_before,
        "i2_before_T_m2": i2_before,
        "i1_after_T_m": i1_after,
        "i2_after_T_m2": i2_after,
        "z_entry_m": z_entry,
        "z_exit_m": z_exit,
        "kick_entry_T_m": kick_entry_tm,
        "kick_exit_T_m": kick_exit_tm,
        "kick_entry_T_mm_for_igor": 1000.0 * kick_entry_tm,
        "kick_exit_T_mm_for_igor": 1000.0 * kick_exit_tm,
        "rms_len_kick_m": rms_len_kick_m,
    }

    return b_corr_t, info


def correct_one_field_component(
    z_m,
    b_t,
    dist_between_kicks_m=1.5,
    rms_len_kick_mm=100.0,
    kick_center_m=None,
):
    """
    Correct one magnetic-field component by adding two Gaussian end kicks.

    This is the modern Python analog of the old Igor/SRW routine:

        SrwUndCorrectFieldInt

    Parameters
    ----------
    z_m : array_like
        Longitudinal coordinate [m].
    b_t : array_like
        Magnetic field component [T].
    dist_between_kicks_m : float
        Distance between entrance and exit correction kicks [m].
    rms_len_kick_mm : float
        RMS length of each Gaussian correction kick [mm].
    kick_center_m : float or None
        Center of the two-kick system [m]. If None, use center of z_m.

    Returns
    -------
    b_corr_t : ndarray
        Corrected magnetic field [T].
    info : dict
        Diagnostics containing pre/post integrals and kick strengths.
    """
    z_m = np.asarray(z_m, dtype=float)
    b_t = np.asarray(b_t, dtype=float)

    if z_m.ndim != 1:
        raise ValueError("z_m must be one-dimensional.")

    if b_t.shape != z_m.shape:
        raise ValueError("b_t must have the same shape as z_m.")

    if dist_between_kicks_m <= 0.0:
        raise ValueError("dist_between_kicks_m must be positive.")

    if rms_len_kick_mm <= 0.0:
        raise ValueError("rms_len_kick_mm must be positive.")

    if kick_center_m is None:
        kick_center_m = 0.5 * (z_m[0] + z_m[-1])

    rms_len_kick_m = 1e-3 * rms_len_kick_mm

    z_entry = kick_center_m - 0.5 * dist_between_kicks_m
    z_exit = kick_center_m + 0.5 * dist_between_kicks_m

    g_entry = normalized_gaussian(z_m, z_entry, rms_len_kick_m)
    g_exit = normalized_gaussian(z_m, z_exit, rms_len_kick_m)

    i1_before, i2_before = field_integrals(z_m, b_t)

    g_entry_i1, g_entry_i2 = field_integrals(z_m, g_entry)
    g_exit_i1, g_exit_i2 = field_integrals(z_m, g_exit)

    response = np.array(
        [
            [g_entry_i1, g_exit_i1],
            [g_entry_i2, g_exit_i2],
        ],
        dtype=float,
    )

    target = -np.array([i1_before, i2_before], dtype=float)

    kick_entry_tm, kick_exit_tm = np.linalg.solve(response, target)

    b_corr_t = b_t + kick_entry_tm * g_entry + kick_exit_tm * g_exit

    i1_after, i2_after = field_integrals(z_m, b_corr_t)

    info = {
        "i1_before_T_m": i1_before,
        "i2_before_T_m2": i2_before,
        "i1_after_T_m": i1_after,
        "i2_after_T_m2": i2_after,
        "z_entry_m": z_entry,
        "z_exit_m": z_exit,
        "kick_entry_T_m": kick_entry_tm,
        "kick_exit_T_m": kick_exit_tm,
        "rms_len_kick_m": rms_len_kick_m,
    }

    return b_corr_t, info
