import numpy as np
from pygaia.errors.spectroscopic import radial_velocity_uncertainty
from pygaia.errors.astrometric import parallax_uncertainty
from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
from gaiaunlimited.scanninglaw import GaiaScanningLaw
from cluster_sampler import ClusterSampler
from PhotoUncertainty import IRPhotoUncertainty, Edr3LogMagUncertainty


class PhotHandler:
    def __init__(self, cluster_object, spline_csv, errors_outside_range):
        self.cluster_object = cluster_object
        print('Initializing the scanning law object...')
        # self.sl = GaiaScanningLaw('dr3_nominal')
        # self.n_obs = self.query_nobs()
        self.n_obs = None
        # GAIA apparent magnitude uncertainties
        self.spline_csv = spline_csv
        self.u = None
        self.g_mag_err = None
        self.bp_mag_err = None
        self.rp_mag_err = None
        # IR apparent magnitude uncertainties
        self.u_ir = IRPhotoUncertainty(errors_outside_range=errors_outside_range)
        self.j_mag_err = None
        self.h_mag_err = None
        self.ks_mag_err = None
        self.irac1_mag_err = None
        self.irac2_mag_err = None
        self.irac3_mag_err = None
        self.irac4_mag_err = None
        self.mips1_mag_err = None
        self.w1_mag_err = None
        self.w2_mag_err = None
        self.w3_mag_err = None
        self.w4_mag_err = None
        # Compute uncertainties
        self.compute_uncertainties_phot()

    def compute_uncertainties_phot(self):
        # Gaia uncertainties
        self.u = Edr3LogMagUncertainty(self.spline_csv, self.n_obs)
        self.g_mag_err = self.u.log_mag_err('g', self.g_mag())
        self.bp_mag_err = self.u.log_mag_err('bp', self.bp_mag())
        self.rp_mag_err = self.u.log_mag_err('rp', self.rp_mag())
        # Apply simple completeness to the Gaia uncertainties
        self.g_mag_err[(self.g_mag() < 4) | (self.g_mag() > 20.)] = np.nan
        self.bp_mag_err[(self.bp_mag() < 0) | (self.bp_mag() > 21.)] = np.nan
        self.rp_mag_err[(self.rp_mag() < 0) | (self.rp_mag() > 19.)] = np.nan
        # IR uncertainties
        self.j_mag_err = self.u_ir.get_sigma('J', self.j_mag())
        self.h_mag_err = self.u_ir.get_sigma('H', self.h_mag())
        self.ks_mag_err = self.u_ir.get_sigma('Ks', self.ks_mag())
        self.irac1_mag_err = self.u_ir.get_sigma('IRAC-1', self.irac1_mag())
        self.irac2_mag_err = self.u_ir.get_sigma('IRAC-2', self.irac2_mag())
        self.irac3_mag_err = self.u_ir.get_sigma('IRAC-3', self.irac3_mag())
        self.irac4_mag_err = self.u_ir.get_sigma('IRAC-4', self.irac4_mag())
        self.mips1_mag_err = self.u_ir.get_sigma('MIPS-1', self.mips1_mag())
        self.w1_mag_err = self.u_ir.get_sigma('W1', self.w1_mag())
        self.w2_mag_err = self.u_ir.get_sigma('W2', self.w2_mag())
        self.w3_mag_err = self.u_ir.get_sigma('W3', self.w3_mag())
        self.w4_mag_err = self.u_ir.get_sigma('W4', self.w4_mag())
        # Add completeness to the IR uncertainties: 2MASS
        missing_j_val = self.u_ir.completeness_ir('J', self.j_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.j_mag_err[missing_j_val] = np.nan
        missing_h_val = self.u_ir.completeness_ir('H', self.h_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.h_mag_err[missing_h_val] = np.nan
        missing_ks_val = self.u_ir.completeness_ir('Ks', self.ks_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.ks_mag_err[missing_ks_val] = np.nan
        # Add completeness to the IR uncertainties: WISE
        missing_w1_val = self.u_ir.completeness_ir('W1', self.w1_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.w1_mag_err[missing_w1_val] = np.nan
        missing_w2_val = self.u_ir.completeness_ir('W2', self.w2_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.w2_mag_err[missing_w2_val] = np.nan
        missing_w3_val = self.u_ir.completeness_ir('W3', self.w3_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.w3_mag_err[missing_w3_val] = np.nan
        missing_w4_val = self.u_ir.completeness_ir('W4', self.w4_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.w4_mag_err[missing_w4_val] = np.nan
        # Add completeness to the IR uncertainties: Spitzer
        missing_irac1_val = self.u_ir.completeness_ir('IRAC-1', self.irac1_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.irac1_mag_err[missing_irac1_val] = np.nan
        missing_irac2_val = self.u_ir.completeness_ir('IRAC-2', self.irac2_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.irac2_mag_err[missing_irac2_val] = np.nan
        missing_irac3_val = self.u_ir.completeness_ir('IRAC-3', self.irac3_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.irac3_mag_err[missing_irac3_val] = np.nan
        missing_irac4_val = self.u_ir.completeness_ir('IRAC-4', self.irac4_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.irac4_mag_err[missing_irac4_val] = np.nan
        missing_mips1_val = self.u_ir.completeness_ir('MIPS-1', self.mips1_mag()) < np.random.uniform(0, 1, self.n_obs)
        self.mips1_mag_err[missing_mips1_val] = np.nan
        return

    def M_G(self):
        return self.cluster_object.data_phot['M_G'].values

    def M_Gbp(self):
        return self.cluster_object.data_phot['G_BP'].values

    def M_Grp(self):
        return self.cluster_object.data_phot['G_RP'].values

    def M_J(self):
        return self.cluster_object.data_phot['J'].values

    def M_H(self):
        return self.cluster_object.data_phot['H'].values

    def M_Ks(self):
        return self.cluster_object.data_phot['Ks'].values

    def M_IRAC1(self):
        return self.cluster_object.data_phot['IRAC-1'].values

    def M_IRAC2(self):
        return self.cluster_object.data_phot['IRAC-2'].values

    def M_IRAC3(self):
        return self.cluster_object.data_phot['IRAC-3'].values

    def M_IRAC4(self):
        return self.cluster_object.data_phot['IRAC-4'].values

    def M_MIPS1(self):
        return self.cluster_object.data_phot['MIPS-1'].values

    def M_W1(self):
        return self.cluster_object.data_phot['W1'].values

    def M_W2(self):
        return self.cluster_object.data_phot['W2'].values

    def M_W3(self):
        return self.cluster_object.data_phot['W3'].values

    def M_W4(self):
        return self.cluster_object.data_phot['W4'].values

    def teff(self):
        return self.cluster_object.data_phot['teff'].values

    def logg(self):
        return self.cluster_object.data_phot['logg'].values

    def mass(self):
        return self.cluster_object.data_phot['mass'].values

    def lifetime(self):
        return self.cluster_object.data_phot['lifetime_logAge'].values

    def is_binary(self):
        return self.cluster_object.data_phot['is_binary'].values

    def apparent_mag(self, M):
        distance = self.cluster_object.skycoord.distance.value
        return M + 5 * np.log10(distance) - 5

    def g_mag(self):
        return self.apparent_mag(self.M_G())

    def bp_mag(self):
        return self.apparent_mag(self.M_Gbp())

    def rp_mag(self):
        return self.apparent_mag(self.M_Grp())

    def j_mag(self):
        return self.apparent_mag(self.M_J())

    def h_mag(self):
        return self.apparent_mag(self.M_H())

    def ks_mag(self):
        return self.apparent_mag(self.M_Ks())

    def irac1_mag(self):
        return self.apparent_mag(self.M_IRAC1())

    def irac2_mag(self):
        return self.apparent_mag(self.M_IRAC2())

    def irac3_mag(self):
        return self.apparent_mag(self.M_IRAC3())

    def irac4_mag(self):
        return self.apparent_mag(self.M_IRAC4())

    def mips1_mag(self):
        return self.apparent_mag(self.M_MIPS1())

    def w1_mag(self):
        return self.apparent_mag(self.M_W1())

    def w2_mag(self):
        return self.apparent_mag(self.M_W2())

    def w3_mag(self):
        return self.apparent_mag(self.M_W3())

    def w4_mag(self):
        return self.apparent_mag(self.M_W4())

    # def query_nobs(self):
    #     ra = self.cluster_object.skycoord.ra.value
    #     dec = self.cluster_object.skycoord.dec.value
    #
    #     # Define helper function
    #     def get_totaln(*args):
    #         return sum(self.sl.query(*args, count_only=True))
    #
    #     # Query the number of observations
    #     n_obs = [get_totaln(*args) for args in zip(ra, dec)]
    #     return np.array(n_obs)


class UncertaintyHandler(PhotHandler):
    def __init__(self, cluster_object: ClusterSampler, spline_csv: str,
                 release: str = 'dr3', errors_outside_range='nan'):
        """Takes effective temperature, surface gravity, absolute G, BP, RP magnitudes, and the distance.
        Computes parallax, radial velocity, and proper motion uncertainties.
        See: https://www.cosmos.esa.int/web/gaia/science-performance#spectroscopic%20performance
        """
        super().__init__(cluster_object, spline_csv, errors_outside_range)
        self.release = release
        # G_RVS maginitude
        self.g_rvs = self.compute_grvs(self.g_mag(), self.rp_mag())
        # Gaia uncertainties to compute
        self.ra_err = None
        self.dec_err = None
        self.plx_err = None
        self.pmra_err = None
        self.pmdec_err = None
        self.rv_err = None
        self.completeness_gaia = None
        # Compute uncertainties
        self.compute_uncertainties_astrometry()
        # Estimate completeness
        print('Estimating completeness...')
        self.estimate_completeness()

    def new_cluster(self, cluster_object):
        self.cluster_object = cluster_object
        # self.n_obs = self.query_nobs()
        self.n_obs = None
        self.compute_uncertainties_phot()
        self.compute_uncertainties_astrometry()
        self.estimate_completeness()
        return

    def estimate_completeness(self):
        """Estimate the completeness of the cluster"""
        # Compute the completeness
        mapHpx7 = DR3SelectionFunctionTCG()
        self.completeness_gaia = mapHpx7.query(self.cluster_object.skycoord, self.g_mag())
        # Completeness on the bright end
        c_bright = lambda x: 0.025 * x + 0.7
        cut = (self.g_mag() < 12) & (self.g_mag() >= 4)
        self.completeness_gaia[cut] = c_bright(self.g_mag()[cut])
        return

    def compute_parallax_uncertainty(self):
        """Compute parallax uncertainty in mas
        Function parallax_uncertainty returns paralax uncertainty in µas
        """
        if self.n_obs is None:
            self.plx_err = parallax_uncertainty(self.g_mag(), release=self.release) / 1_000.
        else:
            __nobs_baseline_plx = 200
            __nobs_astrometric_good = 8.3 * self.n_obs
            n_obs_relative = __nobs_astrometric_good / __nobs_baseline_plx
            self.plx_err = parallax_uncertainty(self.g_mag(), release=self.release) / 1_000. / np.sqrt(n_obs_relative)
        return

    def compute_uncertainties_astrometry(self):
        """Compute uncertainties for the astrometric parameters
        For the conversion factors see:
            https://www.cosmos.esa.int/web/gaia/science-performance#spectroscopic%20performance
        """
        self.compute_parallax_uncertainty()
        self.ra_err = 0.8 * self.plx_err
        self.dec_err = 0.7 * self.plx_err
        self.pmra_err = 1.03 * self.plx_err
        self.pmdec_err = 0.89 * self.plx_err
        self.rv_err = radial_velocity_uncertainty(self.g_mag(), self.teff(), self.logg(), release=self.release)
        return

    @staticmethod
    def compute_grvs(G, Grp):
        g_rp = G - Grp
        f1_g_rvs = -0.0397 - 0.2852 * g_rp - 0.033 * g_rp ** 2 - 0.0867 * g_rp ** 3
        f2_g_rvs = -4.0618 + 10.0187 * g_rp - 9.0532 * g_rp ** 2 + 2.6089 * g_rp ** 3
        # functions valid within the following ranges
        range_1 = g_rp < 1.2
        range_2 = 1.2 <= g_rp
        # Compute G_RVS
        if isinstance(G, np.ndarray):
            grvs = np.zeros_like(g_rp)
            grvs[range_1] = f1_g_rvs[range_1] + Grp[range_1]
            grvs[range_2] = f2_g_rvs[range_2] + Grp[range_2]
            return grvs
        else:
            if range_1:
                return f1_g_rvs + Grp
            else:
                return f2_g_rvs + Grp
