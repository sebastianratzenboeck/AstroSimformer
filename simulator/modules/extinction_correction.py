import numpy as np
# Correction coefficients from PARSEC website


def gaia_extinction(A_V, M_G, G_BP, G_RP):
    """Apply extinction to individual isochrone"""
    # Define extinction correction coefficients
    corr_Gmag = 0.83627
    corr_BPmag = 1.08337
    corr_RPmag = 0.63439
    # Compute extincted magnitudes
    M_G_ext = M_G + A_V * corr_Gmag
    G_BP_ext = G_BP + A_V * corr_BPmag
    G_RP_ext = G_RP + A_V * corr_RPmag
    return M_G_ext, G_BP_ext, G_RP_ext


def wise_extinction(A_V, W1, W2, W3, W4):
    """Apply extinction to individual isochrone"""
    # Define extinction correction coefficients
    corr_W1 = 0.05688
    corr_W2 = 0.03427
    corr_W3 = 0.00707
    corr_W4 = 0.00274
    # Compute extincted magnitudes
    W1_ext = W1 + A_V * corr_W1
    W2_ext = W2 + A_V * corr_W2
    W3_ext = W3 + A_V * corr_W3
    W4_ext = W4 + A_V * corr_W4
    return W1_ext, W2_ext, W3_ext, W4_ext


def tmass_extinction(A_V, J, H, Ks):
    """Apply extinction to individual isochrone"""
    # Define extinction correction coefficients
    corr_J = 0.28665
    corr_H = 0.18082
    corr_Ks = 0.11675
    # Compute extincted magnitudes
    J_ext = J + A_V * corr_J
    H_ext = H + A_V * corr_H
    Ks_ext = Ks + A_V * corr_Ks
    return J_ext, H_ext, Ks_ext


def spitzer_extinction(A_V, IRAC1, IRAC2, IRAC3, IRAC4, MIPS1):
    corr_irac1 = 0.05228
    corr_irac2 = 0.03574
    corr_irac3 = 0.02459
    corr_irac4 = 0.01433
    corr_mips1 = 0.00245
    # Compute extincted magnitudes
    IRAC1_ext = IRAC1 + A_V * corr_irac1
    IRAC2_ext = IRAC2 + A_V * corr_irac2
    IRAC3_ext = IRAC3 + A_V * corr_irac3
    IRAC4_ext = IRAC4 + A_V * corr_irac4
    MIPS1_ext = MIPS1 + A_V * corr_mips1
    return IRAC1_ext, IRAC2_ext, IRAC3_ext, IRAC4_ext, MIPS1_ext


class ExtinctionCorrection:
    def __init__(self, data_phot):
        self.data_phot = data_phot

    def apply_extinction(self, A_V, gaia=True, wise=True, tmass=True, spitzer=True):
        if gaia:
            self.data_phot['M_G'], self.data_phot['G_BP'], self.data_phot['G_RP'] = gaia_extinction(
                A_V,
                self.data_phot['M_G'].values, self.data_phot['G_BP'].values, self.data_phot['G_RP'].values
            )
        if wise:
            self.data_phot['W1'], self.data_phot['W2'], self.data_phot['W3'], self.data_phot['W4'] = wise_extinction(
                A_V,
                self.data_phot['W1'].values, self.data_phot['W2'].values,
                self.data_phot['W3'].values, self.data_phot['W4'].values
            )
        if tmass:
            self.data_phot['J'], self.data_phot['H'], self.data_phot['Ks'] = tmass_extinction(
                A_V,
                self.data_phot['J'].values, self.data_phot['H'].values, self.data_phot['Ks'].values
            )
        if spitzer:
            self.data_phot['IRAC-1'], self.data_phot['IRAC-2'], self.data_phot['IRAC-3'], self.data_phot['IRAC-4'], self.data_phot['MIPS-1'] = spitzer_extinction(
                A_V,
                self.data_phot['IRAC-1'].values, self.data_phot['IRAC-2'].values,
                self.data_phot['IRAC-3'].values, self.data_phot['IRAC-4'].values,
                self.data_phot['MIPS-1'].values
            )
        return self.data_phot
