import copy
import numpy as np
import pandas as pd
from utils.compute_uncertainties import UncertaintyHandler


class ErrorBase:
    def __init__(self, unc_obj: UncertaintyHandler, astrometric_fname: str):
        self.unc_obj = unc_obj
        self.astrometric_mean = None
        self.astrometric_cov = None
        self.astrometric_corr_features = None
        # Load the astrometric parameters
        self.set_astrometric_params(astrometric_fname)
        # Define the columns for the astrometric and radial velocity parameters
        self.astrometric_errors = [
            "ra_error", "dec_error", "parallax_error", "pmra_error", "pmdec_error", "radial_velocity_error"
        ]
        # Define the astrometric features and errors
        self.features_gaia = [
            "ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity",
            "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"
        ]
        self.features_ir = [
            "j", "h", "k", "w1", "w2", "w3", "w4",
            "irac1", "irac2", "irac3", "irac4", "mips1"
        ]
        self.features = self.features_gaia + self.features_ir
        self.not_measured = {
            'ra': np.isnan(self.unc_obj.ra_err),
            'dec': np.isnan(self.unc_obj.dec_err),
            'parallax': np.isnan(self.unc_obj.plx_err),
            'pmra': np.isnan(self.unc_obj.pmra_err),
            'pmdec': np.isnan(self.unc_obj.pmdec_err),
            'radial_velocity': np.isnan(self.unc_obj.rv_err),
            'phot_g_mean_mag': np.isnan(self.unc_obj.g_mag_err),
            'phot_bp_mean_mag': np.isnan(self.unc_obj.bp_mag_err),
            'phot_rp_mean_mag': np.isnan(self.unc_obj.rp_mag_err),
            'j': np.isnan(self.unc_obj.j_mag_err),
            'h': np.isnan(self.unc_obj.h_mag_err),
            'k': np.isnan(self.unc_obj.ks_mag_err),
            'w1': np.isnan(self.unc_obj.w1_mag_err),
            'w2': np.isnan(self.unc_obj.w2_mag_err),
            'w3': np.isnan(self.unc_obj.w3_mag_err),
            'w4': np.isnan(self.unc_obj.w4_mag_err),
            'irac1': np.isnan(self.unc_obj.irac1_mag_err),
            'irac2': np.isnan(self.unc_obj.irac2_mag_err),
            'irac3': np.isnan(self.unc_obj.irac3_mag_err),
            'irac4': np.isnan(self.unc_obj.irac4_mag_err),
            'mips1': np.isnan(self.unc_obj.mips1_mag_err)
        }
        # Correlation positions in correlation/covariance matrix
        self.corr_map = {
            "ra_dec_corr": [0, 1],
            "ra_parallax_corr": [0, 2],
            "ra_pmra_corr": [0, 3],
            "ra_pmdec_corr": [0, 4],
            "dec_parallax_corr": [1, 2],
            "dec_pmra_corr": [1, 3],
            "dec_pmdec_corr": [1, 4],
            "parallax_pmra_corr": [2, 3],
            "parallax_pmdec_corr": [2, 4],
            "pmra_pmdec_corr": [3, 4],
        }
        self.X = self.set_X()
        # Prepare cov. matrix and Cholesky decomposition storage
        self.X_corr = None  # error features
        self.C = None  # covariance matrix for "astrometric_features" features
        self.L = None  # Cholesky decomposition, needed for fast sampling with numba
        self.simulate_errors()

    def set_X(self):
        c = self.unc_obj.cluster_object.skycoord
        ra, dec, dist = c.ra.value, c.dec.value, c.distance.value
        cos_dec = np.cos(np.radians(dec))
        pmra, pmdec, rv = c.pm_ra.value * cos_dec, c.pm_dec.value, c.radial_velocity.value
        parallax = 1000 / dist
        # Set the astrometric features + magnitudes
        X = np.vstack([
            ra, dec, parallax, pmra, pmdec, rv,
            self.unc_obj.g_mag(),
            self.unc_obj.bp_mag(),
            self.unc_obj.rp_mag(),
            self.unc_obj.j_mag(),
            self.unc_obj.h_mag(),
            self.unc_obj.ks_mag(),
            self.unc_obj.w1_mag(),
            self.unc_obj.w2_mag(),
            self.unc_obj.w3_mag(),
            self.unc_obj.w4_mag(),
            self.unc_obj.irac1_mag(),
            self.unc_obj.irac2_mag(),
            self.unc_obj.irac3_mag(),
            self.unc_obj.irac4_mag(),
            self.unc_obj.mips1_mag()
        ]).T
        return X

    def set_astrometric_params(self, fname):
        npzfile = np.load(fname)
        self.astrometric_mean = npzfile['mu']
        self.astrometric_cov = npzfile['cov']
        self.astrometric_corr_features = npzfile['features']

    def simulate_astrometric_cov(self, n_samples):
        X = np.random.multivariate_normal(self.astrometric_mean, self.astrometric_cov, n_samples)
        df = pd.DataFrame(X, columns=self.astrometric_corr_features)
        return df

    def run_checks(self):
        eps = 1.e-4
        # Correlations need to be in [-1, 1]
        for col in self.corr_map.keys():
            self.X_corr.loc[self.X_corr[col] > 1, col] = 1 - eps
            self.X_corr.loc[self.X_corr[col] < -1, col] = -1 + eps
        return

    def simulate_errors(self):
        n_samples = self.X.shape[0]
        self.X_corr = self.simulate_astrometric_cov(n_samples)
        # return astrometric_errors, rv_errors
        self.run_checks()
        self.build_covariance_matrix()
        return

    def build_covariance_matrix(self):
        """Create covariance matrix from input features"""
        # Start building covariance matrix
        nb_sources, nb_features = self.X.shape
        self.C = np.zeros((nb_sources, nb_features, nb_features), dtype=np.float32)
        diag = np.arange(nb_features)
        binary_err_rv = self.unc_obj.is_binary() * np.random.normal(5, 4, nb_sources)
        binary_err_pm = self.unc_obj.is_binary() * np.random.normal(3, 4, nb_sources)
        binary_err_rv[binary_err_rv < 0] = 0.
        binary_err_pm[binary_err_pm < 0] = 0.
        self.C[:, 0, 0] = self.unc_obj.ra_err
        self.C[:, 1, 1] = self.unc_obj.dec_err
        self.C[:, 2, 2] = self.unc_obj.plx_err
        self.C[:, 3, 3] = self.unc_obj.pmra_err + binary_err_pm
        self.C[:, 4, 4] = self.unc_obj.pmdec_err + binary_err_pm
        self.C[:, 5, 5] = self.unc_obj.rv_err + binary_err_rv
        self.C[:, 6, 6] = self.unc_obj.g_mag_err
        self.C[:, 7, 7] = self.unc_obj.bp_mag_err
        self.C[:, 8, 8] = self.unc_obj.rp_mag_err
        # Add IR photometry
        self.C[:, 9, 9] = self.unc_obj.j_mag_err
        self.C[:, 10, 10] = self.unc_obj.h_mag_err
        self.C[:, 11, 11] = self.unc_obj.ks_mag_err
        # Keep WISE photometric erros small
        wise_err_scale = 1.
        self.C[:, 12, 12] = self.unc_obj.w1_mag_err * wise_err_scale
        self.C[:, 13, 13] = self.unc_obj.w2_mag_err * wise_err_scale
        self.C[:, 14, 14] = self.unc_obj.w3_mag_err * wise_err_scale
        self.C[:, 15, 15] = self.unc_obj.w4_mag_err * wise_err_scale
        self.C[:, 16, 16] = self.unc_obj.irac1_mag_err
        self.C[:, 17, 17] = self.unc_obj.irac2_mag_err
        self.C[:, 18, 18] = self.unc_obj.irac3_mag_err
        self.C[:, 19, 19] = self.unc_obj.irac4_mag_err
        self.C[:, 20, 20] = self.unc_obj.mips1_mag_err
        # Remove nans
        self.C = np.nan_to_num(self.C, nan=1e3, copy=True)
        # Squre variance -> std dev
        self.C[:, diag, diag] = self.C[:, diag, diag] ** 2
        # ----- Fill in off-diagonal elements -----
        for column, (i, j) in self.corr_map.items():
            self.C[:, i, j] = self.X_corr[column].fillna(0).to_numpy(dtype=np.float32).ravel()
            self.C[:, i, j] *= (
                    self.C[:, i, i] * self.C[:, j, j]
            )  # transform correlation to covariance
            self.C[:, j, i] = self.C[:, i, j]  # fill in symmetric component
        # Compute Cholesky decomposition
        epsilon = 0.0001  # Define epsilon to add small pertubation to
        self.L = copy.deepcopy(self.C)  # .copy()
        # add small pertubation to covariance matrix, because its eigenvalues can decay
        # very rapidly and without this stabilization the Cholesky decomposition fails
        self.L[:, diag, diag] += epsilon
        #  Cholesky decomposition:
        for k in range(nb_sources):
            # set i'th element to Cholensky decomposition of covariance matrix
            try:
                self.L[k, :, :] = np.linalg.cholesky(self.L[k, :, :])
            # except np.linalg.LinAlgError:
            except:
                # Make positive definite
                # min_ev = np.linalg.eigvals(self.L[k, :, :]).min()
                # C_new = self.L[k, :, :] + (-min_ev + epsilon) * np.eye(nb_features)
                # self.L[k, :, :] = np.linalg.cholesky(C_new)
                # Just save the diagonal elements
                C_new = np.diag(np.diag(self.C[k, :, :]))
                self.L[k, :, :] = np.linalg.cholesky(C_new)

    def convolve(self):
        nb_points, nb_covfeats = self.X.shape[0], len(self.features)
        if nb_points != self.L.shape[0]:
            raise ValueError("Number of points in X and L do not match")

        add2X = self.__new_sample(self.L, nb_points, nb_covfeats)
        X_new = self.X + np.asarray(add2X)
        # Check if the declination angle is out of bounds, if so we fix that
        X_new[:, 0], X_new[:, 1] = self.transform_to_radec_bounds(
            ra=X_new[:, 0], dec=X_new[:, 1]
        )
        df_new = pd.DataFrame(X_new, columns=self.features)
        return self.post_process(df_new)

    def post_process(self, df):
        # Simulate missing values
        for col, isna in self.not_measured.items():
            df.loc[isna, col] = np.nan
        # Remove stars based on completeness estimate
        rand_nb = np.random.uniform(0, 1, df.shape[0])
        is_complete = rand_nb < self.unc_obj.completeness_gaia
        # Remove stars lifetimes less than the cluster age
        is_alive = self.unc_obj.lifetime() > self.unc_obj.cluster_object.cluster_logAge
        df.loc[~is_complete, self.features_gaia] = np.nan
        return df.loc[is_alive]

    @staticmethod
    def __new_sample(L, nb_points, nb_covfeats):
        """Sample a single data point from normal distribution
        Here we calculate the distance we have to travel away from data point:
        X'[i] = X[i] + sample_list[i]
        """
        sample_list = list()
        for i in range(nb_points):
            u = np.random.normal(loc=0, scale=1, size=nb_covfeats).astype(np.float32)
            mult = L[i] @ u
            sample_list.append(mult)
        return sample_list

    def transform_to_radec_bounds(self, ra, dec):
        # skycoords can deal with ra>360 or ra<0, but not with dec out of [-90, 90]
        dec_under_range = dec < -90.0
        dec_over_range = dec > 90.0
        # Mirror the declination angle
        if np.any(dec_under_range):
            dec[dec_under_range] += 2 * (np.abs(dec[dec_under_range]) - 90)
            ra[dec_under_range] += 180
        if np.any(dec_over_range):
            dec[dec_over_range] -= 2 * (np.abs(dec[dec_over_range]) - 90)
            ra[dec_over_range] += 180
        # Correct ra if it is out of bounds
        ra_over_range = ra > 360
        ra_under_range = ra < 0
        if np.any(ra_over_range):
            ra[ra_over_range] -= 360
        if np.any(ra_under_range):
            ra[ra_under_range] += 360
        return ra, dec