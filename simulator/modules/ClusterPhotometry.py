import imf
import numpy as np
import pandas as pd
from isochrones import ParsecGaia, ParsecIR
from extinction_correction import ExtinctionCorrection


class ClusterPhotometry(ExtinctionCorrection):
    def __init__(self, parsec_gaia_folder, parsec_ir_folder):
        super().__init__(None)
        self.p_gaia_obj = ParsecGaia(parsec_gaia_folder)
        self.p_ir_obj = ParsecIR(parsec_ir_folder)
        self.source_data_photometry = None
        self.data_photometry_binaries = None

    def make_photometry(self, cluster_mass, logAge, feh, A_V, f_bin_total=0.5):
        mass_samples = imf.make_cluster(cluster_mass)
        mass_samples = np.sort(mass_samples)
        logAge_samples = np.full_like(mass_samples, logAge)
        feh_samples = np.full_like(mass_samples, feh)
        # Get parsec Gaia photometry
        df_p_gaia = self.p_gaia_obj.query_cmd(mass_samples, logAge_samples, feh_samples)
        # df_p_gaia['mass'] = mass_samples
        # Get parsec IR photometry
        df_p_ir = self.p_ir_obj.query_cmd(mass_samples, logAge_samples, feh_samples)
        df_p_ir['mass'] = mass_samples

        self.source_data_photometry = pd.concat([
            df_p_gaia[self.p_gaia_obj.predict_clean_names[:-2]],
            df_p_ir[self.p_ir_obj.predict_clean_names + ['mass']]
        ], axis=1)
        if f_bin_total > 0:
            self.add_binaries(f_bin_total)
            self.data_phot = self.data_photometry_binaries
        else:
            self.data_phot = self.source_data_photometry
        self.compute_lifetimes()
        # Add extinction
        self.apply_extinction(A_V)
        return self.data_phot

    @staticmethod
    def add_magnitudes(*args):
        return -2.5 * np.log10(np.sum([10 ** (-0.4 * M_i) for M_i in args], axis=0))

    def create_binaries_pairs(self, f_bin_total):
        n = self.source_data_photometry.shape[0]
        random_idx_pairs = np.array([[i, j] for i, j in zip(np.arange(n), np.random.permutation(n))])
        rand_pairs_boolarr = np.random.uniform(0, 1, n) < f_bin_total
        random_idx_pairs_filtered = random_idx_pairs[rand_pairs_boolarr]
        # Sort by joint mass
        random_idx_pairs_filtered = random_idx_pairs_filtered[random_idx_pairs_filtered.sum(axis=1).argsort()][::-1]
        all_pairs_final = []
        unique_sources = set()
        for i, j in random_idx_pairs_filtered:
            if (i not in unique_sources) and (j not in unique_sources):
                all_pairs_final.append([i, j])
                unique_sources.add(i)
                unique_sources.add(j)
        return np.array(all_pairs_final)

    def add_binaries(self, f_bin_total):
        all_pairs_final = self.create_binaries_pairs(f_bin_total)
        max_nb = np.max(all_pairs_final, axis=1)
        min_nb = np.min(all_pairs_final, axis=1)
        # Create copy to store binaries in
        self.data_photometry_binaries = self.source_data_photometry.copy()
        # Get photometry of given ids
        df_max = self.data_photometry_binaries.loc[max_nb]
        df_min = self.data_photometry_binaries.loc[min_nb]
        # Compute the combined photometry
        for col in ['M_G', 'G_BP', 'G_RP']:
            self.data_photometry_binaries.loc[max_nb, col] = self.add_magnitudes(
                df_max[col].values, df_min[col].values
            )
        # Compute the combined logg and teff
        for col in ['logg', 'teff']:
            self.data_photometry_binaries.loc[max_nb, col] = np.max(
                np.vstack([df_max[col].values, df_min[col].values]), axis=0
            )
        # Compute combined mass
        self.data_photometry_binaries.loc[max_nb, 'mass'] = df_max['mass'].values  # + df_min['mass'].values
        # Save is binary as columns
        self.data_photometry_binaries['is_binary'] = 0.
        self.data_photometry_binaries.loc[max_nb, 'is_binary'] = 1.
        self.data_photometry_binaries.loc[min_nb, 'is_binary'] = -1.
        # Remove min_nb from the source data
        self.data_photometry_binaries = self.data_photometry_binaries.drop(min_nb)
        return

    def compute_lifetimes(self):
        mass = self.data_phot['mass'].values
        self.data_phot['lifetime_logAge'] = np.log10(10 ** 10 * (1 / mass) ** 2.5)
        return
