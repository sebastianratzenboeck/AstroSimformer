import numpy as np
import pandas as pd
import scipy.interpolate as interpolate


sigma_w1 = {
    # ATTENTION: bright (orig. bright end sigma: 3) and faint end values updated to smaller uncertainties!
    # 'mag': [-2, 3, 3.5, 6.4, 7.3, 7.9, 8.0, 12.0, 13.4, 15.1, 16.3, 19.],  #17.7, 18.4, 19],
    # 'sigma_mag': [2, 2, 1.4, 0.5, 0.2, 0.18, 0.17, 0.26, 0.32, 0.51, 0.94, 2.],  #3.54, 18.2, 30.],
    'mag':       [-2., 0.,  2.,  4.,   6.,   8.,  10.,  12.,  14., 16., 18., 20.],  # 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'sigma_mag': [2.,  1., 0.5, 0.3, 0.04, 0.02, 0.02, 0.02, 0.03, 0.05, 0.2, 2.],
    'range': [-2, 20]
}

sigma_w2 = {
    # ATTENTION: faint end values updated to smaller uncertainties!
    # 'mag': [2, 3.7, 5.9, 6.3, 6.8, 7.2, 10.5, 12.4, 13.0, 14.0, 14.5, 15.1, 17.5],  # 15.9, 16.9, 17.5],
    # 'sigma_mag': [1.5, 0.6, 0.2, 0.14, 0.12, 0.13, 0.19, 0.27, 0.33, 0.5, 0.69, 1.07, 2.],  #2.25, 11.3, 30.],
    'mag':       [0,  2.,   4.,   6.,   8.,  10.,  12.,  14.,  16., 18.],
    'sigma_mag': [2, 0.5, 0.15, 0.02, 0.02, 0.02, 0.02, 0.04, 0.15, 1.8],
    'range': [0, 18]
}

sigma_w3 = {
    # ATTENTION: faint end values updated to smaller uncertainties!
    'mag':       [-6,   -2,   0,    1,     3,  7.3,   8.7,  9.8, 11., 13., 14.],  #10.6, 11.2, 11.8, 12.6, 12.9, 13],
    'sigma_mag': [0.8, 0.5, 0.2, 0.02, 0.013, 0.02, 0.038, 0.06, 0.1, 0.5, 1.4], #1.01, 1.6, 2.87, 6.24, 18.1, 30.],
    'range': [-6, 14]
}

sigma_w4 = {
    # 'mag': [-6, -2, -1, 0.3, 2.8, 4.7, 6.1, 7.1, 7.5, 8.2, 8.9, 9.2, 9.5],
    # ATTENTION: faint end values updated to smaller uncertainties!
    # 'sigma_mag': [0.02, 0.01, 0.005, 0.01, 0.04, 0.14, 0.34, 0.76, 0.8, 0.85, 0.9, 0.95, 2.],  #1.07, 2.2, 4.4, 11.4, 30.],
    'mag':       [-2,     0.,    2.,    4.,    6.,  8., 10.],
    'sigma_mag': [0.001, 0.02, 0.02, 0.025,  0.04, 0.2,  1.],
    'range': [-2, 10]
}

sigma_irac1 = {
    'mag': [5, 5.7, 6.7, 9.0, 9.3, 12.1, 14.2, 15.7, 16.5, 17.2, 17.8, 18.6, 18.9, 20.],
    # ATTENTION: faint end values updated to smaller uncertainties!
    'sigma_mag': [0.002, 0.0005, 0.001, 0.004, 0.0008, 0.003, 0.01, 0.028, 0.05, 0.1, 0.18, 0.53, 0.7, 1.6],  #1.17, 1.6],
    'range': [5, 20]
}

sigma_irac2 = {
    'mag': [4.5, 5.8, 8.3, 8.6, 11.4, 13.2, 14.7, 16., 16.7, 17.5, 18.6, 19.],
    # ATTENTION: faint end values updated to smaller uncertainties!
    'sigma_mag': [0.002, 0.001, 0.004, 0.001, 0.002, 0.008, 0.024, 0.06, 0.13, 0.3, 0.5, 1.], #1.4, 3.],
    'range': [4.5, 19]
}

sigma_irac3 = {
    'mag': [2., 4., 8., 10.5, 11.7, 12.8, 13.9, 14.9, 15.29, 15.6, 16.5],
    # ATTENTION: faint end values updated to smaller uncertainties!
    'sigma_mag': [0.001, 0.0005, 0.002, 0.006, 0.013, 0.029, 0.07, 0.21, 0.49, 0.7, 1],  #8],
    'range': [2, 16.5]
}

sigma_irac4 = {
    'mag': [2.5, 5.6, 7., 9., 10.3, 11.6, 12.5, 13.5, 13.8, 14.8, 15., 16.],
    # ATTENTION: faint end values updated to smaller uncertainties!
    'sigma_mag': [0.001, 0.002, 0.002, 0.003, 0.008, 0.02, 0.04, 0.07, 0.14, 0.48, 0.08, 1],  #7],
    'range': [2.5, 16]
}

sigma_mips1 = {
    'mag': [-2., 0., 6., 7.1, 8.1, 9.1, 9.8, 10.4, 11.],
    'sigma_mag': [0.03, 0.01, 0.015, 0.027, 0.06, 0.13, 0.23, 0.48, 0.5],
    'range': [-2, 11]
}

sigma_j = {
    'mag': [0., 4., 5., 10., 13., 14.5, 15.3, 16.1, 16.8, 17.4, 18.],
    'sigma_mag': [0.25, 0.25, 0.03, 0.024, 0.026, 0.032, 0.046, 0.085, 0.15, 0.25, 0.5],
    'range': [0, 18]
}

sigma_h = {
    'mag': [0., 4., 5., 9.1, 13., 14.5, 15., 15.6, 16.4, 16.9, 18.],
    'sigma_mag': [0.25, 0.25, 0.03, 0.026, 0.026, 0.04, 0.068, 0.126, 0.23, 0.38, 0.5],
    'range': [0., 18]
}

sigma_ks = {
    'mag': [0., 4., 5., 12.2, 13., 14.2, 14.9, 15.6, 16.0, 16.4, 18.],
    'sigma_mag':  [0.25, 0.25, 0.02, 0.02, 0.03, 0.06, 0.12, 0.22, 0.34, 0.47, 0.5],
    'range': [0, 18]
}


coeff_j = {
    'mu_bright': 1,
    'mu_faint': 16.5,
    'width_bright': 1,
    'width_faint': 0.1
}

coeff_h = {
    'mu_bright': 1,
    'mu_faint': 16.0,
    'width_bright': 1,
    'width_faint': 0.1
}

coeff_ks = {
    'mu_bright': 1,
    'mu_faint': 15.6,
    'width_bright': 1,
    'width_faint': 0.1
}

coeff_w1 = {
    'mu_bright': 4,
    'mu_faint': 18,
    'width_bright': 0.6,
    'width_faint': 0.25
}

coeff_w2 = {
    'mu_bright': 3.35,
    'mu_faint': 16.9,
    'width_bright': 0.6,
    'width_faint': 0.2
}

coeff_w3 = {
    'mu_bright': 2.5,
    'mu_faint': 12.5,
    'width_bright': 0.6,
    'width_faint': 0.2
}

coeff_w4 = {
    'mu_bright': -1,
    'mu_faint': 8.7,
    'width_bright': 0.2,
    'width_faint': 0.2
}


def two_sided_sigmoid(x, x_midpoint_left, x_midpoint_right, left_width, right_width):
    left_sigmoid = 1 / (1 + np.exp(-1/left_width * (x - x_midpoint_left)))
    right_sigmoid = 1 / (1 + np.exp(-1/right_width * (x - x_midpoint_right)))
    return np.abs(left_sigmoid - right_sigmoid)


class IRPhotoUncertainty:
    def __init__(self, errors_outside_range=np.nan):
        self.sigma_dict = {
            'W1': sigma_w1,
            'W2': sigma_w2,
            'W3': sigma_w3,
            'W4': sigma_w4,
            'IRAC-1': sigma_irac1,
            'IRAC-2': sigma_irac2,
            'IRAC-3': sigma_irac3,
            'IRAC-4': sigma_irac4,
            'MIPS-1': sigma_mips1,
            'J': sigma_j,
            'H': sigma_h,
            'Ks': sigma_ks
        }
        self.completeness_coeffs = {
            'J': coeff_j,
            'H': coeff_h,
            'Ks': coeff_ks,
            'W1': coeff_w1,
            'W2': coeff_w2,
            'W3': coeff_w3,
            'W4': coeff_w4
        }
        self.errors_outside_range = None
        self.process_extrapolation(errors_outside_range)

    def completeness_ir(self, filter_str, mag):
        comp = np.zeros_like(mag)
        mag_lims = self.sigma_dict[filter_str]['range']
        cut_nonzero = (mag > min(mag_lims)) & (mag < max(mag_lims))
        # Compute the completeness
        if filter_str in ['J', 'H', 'Ks', 'W1', 'W2', 'W3', 'W4']:
            coeffs = self.completeness_coeffs[filter_str]
            comp[cut_nonzero] = two_sided_sigmoid(
                mag[cut_nonzero],
                coeffs['mu_bright'], coeffs['mu_faint'],
                coeffs['width_bright'], coeffs['width_faint']
            )
        else:
            comp[cut_nonzero] = 1
        return comp

    def process_extrapolation(self, errors_outside_range):
        if (errors_outside_range == np.nan) or (errors_outside_range == 'nan'):
            self.errors_outside_range = np.nan
        else:
            self.errors_outside_range = None
        return

    def get_sigma(self, filter_str, mag):
        mag_sigma = np.interp(
            mag, self.sigma_dict[filter_str]['mag'], self.sigma_dict[filter_str]['sigma_mag'],
            left=self.errors_outside_range, right=self.errors_outside_range
        )
        return mag_sigma


class Edr3LogMagUncertainty:
    """Estimate the log(mag) vs mag uncertainty for G, G_BP, G_RP based on Gaia EDR3 photometry."""
    def __init__(self, spline_csv, n_obs=200):
        """Usage
        >>> u = Edr3LogMagUncertainty('LogErrVsMagSpline.csv')
        >>> gmags = np.array([5, 10, 15, 20])
        >>> g200 = u.log_mag_err('g', gmags, 200)
        """
        _df = pd.read_csv(spline_csv)
        splines = dict()
        splines['g'] = self.__init_spline(_df, 'knots_G', 'coeff_G')
        splines['bp'] = self.__init_spline(_df, 'knots_BP', 'coeff_BP')
        splines['rp'] = self.__init_spline(_df, 'knots_RP', 'coeff_RP')
        self.__splines = splines
        self.__nobs_baseline = {'g': 200, 'bp': 20, 'rp': 20}
        self.n_obs = self.set_nobs(n_obs)

    def set_nobs(self, n_obs):
        """Set the number of observations for G, G_BP, G_RP bands.
        The numbers are proportional to the number of matched_transits in Gaia (can be computed via GaiaScanningLaw).
        """
        self.n_obs = None
        if isinstance(n_obs, int):
            if n_obs > 0:
                self.n_obs = {
                    'g': n_obs * 8.5, 'bp': n_obs * 0.95, 'rp': n_obs * 0.91
                }
        elif isinstance(n_obs, np.ndarray):
            self.n_obs = {}
            self.n_obs['g'] = n_obs * 8.5
            self.n_obs['bp'] = n_obs * 0.95
            self.n_obs['rp'] = n_obs * 0.91
        return

    def __init_spline(self, df, col_knots, col_coeff):
        __ddff = df[[col_knots, col_coeff]].dropna()
        return interpolate.BSpline(__ddff[col_knots], __ddff[col_coeff], 3, extrapolate=False)

    def log_mag_err(self, band, mag_val, n_obs=None):
        if n_obs is not None:
            self.set_nobs(n_obs)
        # If number of observations was not passed
        if self.n_obs is None:
            return 10 ** self.__splines[band](mag_val)

        return 10 ** (self.__splines[band](mag_val) - np.log10(
            np.sqrt(self.n_obs[band]) / np.sqrt(self.__nobs_baseline[band])))


