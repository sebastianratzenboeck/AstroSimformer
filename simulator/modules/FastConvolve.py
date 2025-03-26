import numpy as np
import pandas as pd
import imf
from SBI.modules.isochrones import ParsecGaia
from pygaia.errors.spectroscopic import radial_velocity_uncertainty
from pygaia.errors.astrometric import parallax_uncertainty


class FastConvolve:
    def __init__(self, data, logAge, parsec_gaia_folder, release='dr3'):
        """Data is a pandas DataFrame with columns (Gaia Main Database column names)
        see: https://gea.esac.esa.int/archive/documentation/GDR2/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html
        - ra
        - dec
        - parallax
        - pmra
        - pmdec
        - radial_velocity
        """
        self.data = data
        self.N = data.shape[0]
        self.logAge = logAge
        self.parsec_gaia_folder = parsec_gaia_folder
        self.release = release

    def set_data(self, data, logAge):
        self.data = data
        self.N = data.shape[0]
        self.logAge = logAge

    @staticmethod
    def apparent_mag(M, dist):
        return M + 5 * np.log10(dist) - 5

    def assign_photometry(self):
        p_gaia_obj = ParsecGaia(self.parsec_gaia_folder)
        # Get samples from the IMF
        mass_samples = imf.make_cluster(10_000)
        # Choose N random samples from the mass_samples
        mass_samples = np.random.choice(mass_samples, self.N, replace=False)
        mass_samples = np.sort(mass_samples)
        logAge_samples = np.full_like(mass_samples, self.logAge)
        feh_samples = np.full_like(mass_samples, 0.0)
        df_p_ir = p_gaia_obj.query_cmd(mass_samples, logAge_samples, feh_samples)
        # Compute the apparent magnitude
        g_mag = self.apparent_mag(df_p_ir['M_G'].values, 1000 / self.data.parallax.values)
        bp_rp = df_p_ir['G_BP'] - df_p_ir['G_RP']
        g_rp = df_p_ir['M_G'] - df_p_ir['G_RP']
        logg = df_p_ir['logg'].values
        teff = df_p_ir['teff'].values
        return g_mag, bp_rp, g_rp, logg, teff

    def compute_uncertainties_astrometry(self, g_mag, teff, logg):
        """Compute uncertainties for the astrometric parameters
        For the conversion factors see:
            https://www.cosmos.esa.int/web/gaia/science-performance#spectroscopic%20performance
        """
        plx_err = parallax_uncertainty(g_mag, release=self.release) / 1_000.
        ra_err = 0.8 * plx_err
        dec_err = 0.7 * plx_err
        pmra_err = 1.03 * plx_err
        pmdec_err = 0.89 * plx_err
        rv_err = radial_velocity_uncertainty(g_mag, teff, logg, release=self.release)
        return ra_err, dec_err, plx_err, pmra_err, pmdec_err, rv_err

    def compute_errors(self):
        g_mag, bp_rp, g_rp, logg, teff = self.assign_photometry()
        ra_err, dec_err, plx_err, pmra_err, pmdec_err, rv_err = self.compute_uncertainties_astrometry(g_mag, teff, logg)
        return ra_err, dec_err, plx_err, pmra_err, pmdec_err, rv_err, g_mag, bp_rp, g_rp, logg, teff

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

    def new_sample(self, return_dataframe=True):
        ra_err, dec_err, plx_err, pmra_err, pmdec_err, rv_err, g_mag, bp_rp, g_rp, logg, teff = self.compute_errors()
        # Treat astrometric parameters as independent Gaussian variables
        ra_obs = np.random.normal(self.data.ra.values, ra_err)
        dec_obs = np.random.normal(self.data.dec.values, dec_err)
        plx_obs = np.random.normal(self.data.parallax.values, plx_err)
        pmra_obs = np.random.normal(self.data.pmra.values, pmra_err)
        pmdec_obs = np.random.normal(self.data.pmdec.values, pmdec_err)
        rv_obs = np.random.normal(self.data.radial_velocity.values, rv_err)
        # Correct for out of bounds values
        ra_obs, dec_obs = self.transform_to_radec_bounds(ra_obs, dec_obs)
        # Return the new sample
        if return_dataframe:
            # is_seen_by_gaia = ~np.isnan(ra_obs)
            X = np.vstack([
                ra_obs, dec_obs, plx_obs, pmra_obs, pmdec_obs, rv_obs,
                ra_err, dec_err, plx_err, pmra_err, pmdec_err, rv_err,
                g_mag, bp_rp, g_rp, logg, teff
            ]).T
            df = pd.DataFrame(
                X,
                columns=[
                    'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity',
                    'ra_error', 'dec_error', 'parallax_error', 'pmra_error', 'pmdec_error', 'radial_velocity_error',
                    'phot_g_mean_mag', 'bp_rp', 'g_rp', 'logg', 'teff'
                ]
            )
            return df
        else:
            return ra_obs, dec_obs, plx_obs, pmra_obs, pmdec_obs, rv_obs
