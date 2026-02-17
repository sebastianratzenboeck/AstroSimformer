"""
Column definitions for the mock galaxy SimFormer training pipeline.

Shared between prepare_data.py and train_mock_galaxy.py to ensure
consistent column ordering across data preparation and training.
"""

# ---------------------------------------------------------------------------
# Intrinsic stellar parameters (no errors, always observed)
# ---------------------------------------------------------------------------
INTRINSIC_COLS = [
    'glon', 'glat', 'feh', 'm_init', 'logAge', 'rad', 'logL', 'logT', 'logg', 'Av',
]

# ---------------------------------------------------------------------------
# True (noise-free) magnitudes from simulation
# ---------------------------------------------------------------------------
TRUE_MAG_COLS = [
    'GAIA_GAIA3.Gbp_mag', 'GAIA_GAIA3.G_mag', 'GAIA_GAIA3.Grp_mag',
    '2MASS_H_mag', '2MASS_J_mag', '2MASS_Ks_mag',
    'WISE_WISE.W1_mag', 'WISE_WISE.W2_mag',
    'PS1_g_mag', 'PS1_i_mag', 'PS1_r_mag', 'PS1_y_mag', 'PS1_z_mag',
    'CTIO_DECam.g_mag', 'CTIO_DECam.r_mag', 'CTIO_DECam.i_mag',
    'CTIO_DECam.z_mag', 'CTIO_DECam.Y_mag',
]

# ---------------------------------------------------------------------------
# Observed measurements (may be NaN if unobserved)
# ---------------------------------------------------------------------------
OBS_COLS = [
    # Gaia obs (already present in raw data)
    'GAIA_GAIA3.Gbp_mag_obs', 'GAIA_GAIA3.G_mag_obs', 'GAIA_GAIA3.Grp_mag_obs',
    'parallax_obs', # 'distance_obs',
    # Other surveys
    '2MASS_H_mag_obs', '2MASS_J_mag_obs', '2MASS_Ks_mag_obs',
    'WISE_WISE.W1_mag_obs', 'WISE_WISE.W2_mag_obs',
    'PS1_g_mag_obs', 'PS1_i_mag_obs', 'PS1_r_mag_obs', 'PS1_y_mag_obs', 'PS1_z_mag_obs',
    'CTIO_DECam.g_mag_obs', 'CTIO_DECam.r_mag_obs', 'CTIO_DECam.i_mag_obs',
    'CTIO_DECam.z_mag_obs', 'CTIO_DECam.Y_mag_obs',
]

# ---------------------------------------------------------------------------
# Error columns corresponding to OBS_COLS (same order)
# ---------------------------------------------------------------------------
OBS_ERR_COLS = [
    'GAIA_GAIA3.Gbp_mag_err', 'GAIA_GAIA3.G_mag_err', 'GAIA_GAIA3.Grp_mag_err',
    'parallax_err', # 'parallax_err',  # distance_obs uses parallax_err as proxy
    '2MASS_H_mag_err', '2MASS_J_mag_err', '2MASS_Ks_mag_err',
    'WISE_WISE.W1_mag_err', 'WISE_WISE.W2_mag_err',
    'PS1_g_mag_err', 'PS1_i_mag_err', 'PS1_r_mag_err', 'PS1_y_mag_err', 'PS1_z_mag_err',
    'CTIO_DECam.g_mag_err', 'CTIO_DECam.r_mag_err', 'CTIO_DECam.i_mag_err',
    'CTIO_DECam.z_mag_err', 'CTIO_DECam.Y_mag_err',
]

# ---------------------------------------------------------------------------
# Survey definitions for synthetic error generation
# {survey_name: (true_magnitude_columns, fixed_error_sigma)}
# ---------------------------------------------------------------------------
SURVEY_ERRORS = {
    '2MASS': (['2MASS_H_mag', '2MASS_J_mag', '2MASS_Ks_mag'], 0.03),
    'WISE':  (['WISE_WISE.W1_mag', 'WISE_WISE.W2_mag'], 0.05),
    'PS1':   (['PS1_g_mag', 'PS1_i_mag', 'PS1_r_mag', 'PS1_y_mag', 'PS1_z_mag'], 0.02),
    'DECam': (['CTIO_DECam.g_mag', 'CTIO_DECam.r_mag', 'CTIO_DECam.i_mag',
               'CTIO_DECam.z_mag', 'CTIO_DECam.Y_mag'], 0.02),
}

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
ALL_VALUE_COLS = INTRINSIC_COLS + TRUE_MAG_COLS + OBS_COLS
NUM_NODES = len(ALL_VALUE_COLS)
N_INTRINSIC = len(INTRINSIC_COLS)
N_TRUE_MAG = len(TRUE_MAG_COLS)
