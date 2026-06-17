# Params from old Igor files, probably pre-sextupole upgrade era
ALS_OLD = {
    "iavg": 0.5,  # Average current [A]
    "e": 1.9,  # Electron energy [GeV]
    "sig_e": 0.0,  # RMS energy spread
    "sig_x": 0.31e-3,  # Horizontal RMS size [m]
    "sig_x_pr": 0.023e-3,  # Horizontal RMS divergence [rad]
    "sig_y": 0.023e-3,  # Vertical RMS size [m]
    "sig_y_pr": 0.0065e-3,  # Vertical RMS divergence [rad]
}

# Current ring parameters
ALS = {
    "iavg": 0.5,  # Average current [A]
    "e": 1.9,  # Electron energy [GeV]
    "sig_e": 0.95e-3,  # RMS energy spread
    "sig_x": 0.251e-3,  # Horizontal RMS size [m]
    "sig_x_pr": 0.0097e-3,  # Horizontal RMS divergence [rad]
    "sig_y": 0.0083e-3,  # Vertical RMS size [m]
    "sig_y_pr": 0.0048e-3,  # Vertical RMS divergence [rad]
}

# Estimated parameters after the ALS-U upgrade
ALSU = {
    "iavg": 0.5,  # Average current [A]
    "e": 2.0,  # Electron energy [GeV]
    "sig_e": 0.98e-3,  # RMS energy spread
    "sig_x": 0.012e-3,  # Horizontal RMS size [m]
    "sig_x_pr": 0.00575e-3,  # Horizontal RMS divergence [rad]
    "sig_y": 0.014e-3,  # Vertical RMS size [m]
    "sig_y_pr": 0.00493e-3,  # Vertical RMS divergence [rad]
}


def get_beam_params(name):
    if name == "ALS":
        return tuple(ALS.values())
    elif name == "ALSU":
        return tuple(ALSU.values())
    elif name == "ALS_OLD":
        return tuple(ALS_OLD.values())
    else:
        raise ValueError(f"Unknown beam name: {name}")
