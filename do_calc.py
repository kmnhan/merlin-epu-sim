import pathlib

from field import make_field
from spectrum import calculate_spectrum_multi

out_dir = pathlib.Path("simulated")
out_dir.mkdir(exist_ok=True)

EPU_GAP_MM = 44.0
EPU_Z_MM = 0.0
EPU_MODE = "parallel"
CORRECT_FIELD_INTEGRAL = False
CORRECTION_KICK_LENGTH_M = 1.5
CORRECTION_KICK_RMS_LEN_MM = 100.0

FILE_SUFFIX = "_3dfield"


def do_calc():
    field_container = make_field(
        epu_gap=EPU_GAP_MM,
        epu_z=EPU_Z_MM,
        epu_mode=EPU_MODE,
        # x_hw=0.0,
        # y_hw=0.0,
        # z_hw=1.1,
        # nx=1,
        # ny=1,
        # nz=2201,
        x_hw=1.0e-2,
        y_hw=1.0e-2,
        z_hw=1.1,
        nx=5,
        ny=5,
        nz=251,
        correct_field_integral=CORRECT_FIELD_INTEGRAL,
        correction_kick_length_m=CORRECTION_KICK_LENGTH_M,
        correction_kick_rms_len_mm=CORRECTION_KICK_RMS_LEN_MM,
    )

    print("Calculated field")

    file_name = f"traj_gap{EPU_GAP_MM:.0f}_z{EPU_Z_MM:.0f}_{EPU_MODE}"
    file_name += "_corr" if CORRECT_FIELD_INTEGRAL else "_uncorr"
    file_name += FILE_SUFFIX
    file_name += ".h5"

    file_path = (out_dir / file_name).resolve()

    calculate_spectrum_multi(
        field_container=field_container,
        file_path=str(file_path),
        obs_z_m=10.0,  # Observation point z in meters
        theta_x_half_mrad=0.9,  # Observation angle x in mrad
        theta_y_half_mrad=0.9,  # Observation angle y in mrad
        hv_start_eV=1.0,  # Start photon energy in eV
        hv_end_eV=100.0,  # End photon energy in eV
        nhv=250,
        nx=101,
        ny=101,
        n_electrons=100,
        use_mpi=True,  # Whether to use MPI for parallelization.
        extra_attrs={
            "epu_gap_mm": EPU_GAP_MM,
            "epu_z_mm": EPU_Z_MM,
            "epu_mode": EPU_MODE,
            "correct_field_integral": CORRECT_FIELD_INTEGRAL,
            "correction_kick_length_m": CORRECTION_KICK_LENGTH_M,
            "correction_kick_rms_len_mm": CORRECTION_KICK_RMS_LEN_MM,
        },
    )


if __name__ == "__main__":
    do_calc()
