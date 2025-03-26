import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, Galactic
from astropy import units as u
from utils.cluster_sampler import ClusterSampler
from utils.compute_uncertainties import UncertaintyHandler
from utils.compute_observables import ErrorBase


# ------ Path files ------
fpath = '/Users/ratzenboe/Documents/work/code/SF-Retreat2024/'
fname_spline_csv = fpath + 'LogErrVsMagSpline.csv'
fname_astrometric_corr = fpath + 'astrometric_corr.npz'
# -----------------------

class ClusterSim:
    def __init__(self,
                 mu, cov, mass, logAge, feh, A_V, f_bin,
                 parsec_gaia_folder, parsec_ir_folder,
                 spline_csv, astrometric_corr,
                 errors_outside_range='nan'):
        self.spline_csv = spline_csv
        self.astrometric_corr = astrometric_corr
        # Instantiate the ClusterSampler class
        self.cluster_sampler = ClusterSampler(
            mu, cov, mass, logAge, feh, A_V, f_bin, parsec_gaia_folder, parsec_ir_folder
        )
        self.cluster_sampler.simulate_cluster()
        # Instantiate the GaiaUncertainties class and the ErrorBase class
        self.unc_obj = UncertaintyHandler(
            cluster_object=self.cluster_sampler, spline_csv=self.spline_csv,
            errors_outside_range=errors_outside_range
        )
        # Instantiate the ErrorBase class
        self.errs = ErrorBase(unc_obj=self.unc_obj, astrometric_fname=self.astrometric_corr)
        # hardcoded for now, but can be changed to be set by the user it
        self.features_returned = [
            'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
            'phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag', 'M_G', 'X',
            'Y', 'Z', 'U', 'V', 'W'
        ]

    def data_observed(self, **kwargs):
        if kwargs:
            self.cluster_sampler.set_cluster_params(**kwargs)
            self.cluster_sampler.simulate_cluster()
            self.unc_obj.new_cluster(self.cluster_sampler)
            # Instantiate the ErrorBase class
            self.errs = ErrorBase(self.unc_obj, self.astrometric_corr)

        df_obs = self.errs.convolve()
        # Compute observed absolute magnitude
        df_obs['M_G'] = df_obs['phot_g_mean_mag'] - 5 * np.log10(1000 / df_obs['parallax']) + 5
        # XYZ+UVW
        df_cart = self.spher2cart(df_obs[['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity']].values)
        # Add uncertainties in all columns
        is_alive = self.errs.unc_obj.lifetime() > self.errs.unc_obj.cluster_object.cluster_logAge
        N = self.errs.C.shape[1]
        diag_elems = np.arange(N)
        df_unc = pd.DataFrame(
            np.sqrt(self.errs.C[:, diag_elems, diag_elems])[is_alive],
            columns=[col + '_error' for col in self.errs.features]
        )
        df_unc.replace(to_replace=1e3, value=np.nan, inplace=True)
        # Compile observed data
        df_obs = pd.concat([df_obs, df_cart, df_unc], axis=1)
        return df_obs

    def data_true(self):
        df_cart = pd.DataFrame(self.errs.set_X(), columns=self.errs.features)
        df_phot_true = self.cluster_sampler.data_phot.reset_index(drop=True)
        df_Xgal = self.cluster_sampler.X_gal.reset_index(drop=True)
        df_true = pd.concat(
            [df_phot_true, df_Xgal, df_cart], axis=1,
        )
        is_alive = self.errs.unc_obj.lifetime() > self.errs.unc_obj.cluster_object.cluster_logAge
        return df_true.loc[is_alive]  #[self.features_returned + ['logg', 'teff', 'mass', 'is_binary']]
        # return self.cluster_sampler.data_phot

    @staticmethod
    def spher2cart(data):
        ra, dec, parallax, pmra, pmdec, rv = data.T
        dist = 1000 / parallax
        dist[dist < 0] = 1e4
        c = ICRS(
            ra=ra * u.deg, dec=dec * u.deg, distance=dist * u.pc,
            pm_ra_cosdec=pmra * u.mas / u.yr,
            pm_dec=pmdec * u.mas / u.yr,
            radial_velocity=rv * u.km / u.s,
        )
        c = c.transform_to(Galactic())
        c.representation_type = 'cartesian'
        X = np.vstack([c.u.value, c.v.value, c.w.value, c.U.value, c.V.value, c.W.value]).T
        df = pd.DataFrame(X, columns=['X', 'Y', 'Z', 'U', 'V', 'W'])
        return df
