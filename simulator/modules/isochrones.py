import re
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# ----- Corrective factors for extinction correction -----
corr_Gmag = 0.83627
corr_BPmag = 1.08337
corr_RPmag = 0.63439
corr_bprp = corr_BPmag - corr_RPmag
corr_grp = corr_Gmag - corr_RPmag
# --------------------------------------------------------


class ICBase:
    def __init__(self):
        self.data = None
        self.colnames = None
        self.cols_input = None
        self.cols_predict = None
        self.predict_clean_names = None
        self.l_interp = None

    def fit_interpolator(self, n_skip=10):
        df_subset = self.data[::n_skip]
        # Only model mass and age for now
        X = df_subset[self.cols_input].values
        y = df_subset[self.cols_predict].values
        self.l_interp = LinearNDInterpolator(X, y)
        # self.l_interp = NearestNDInterpolator(X, y)

    def query_cmd(self, mass, age, feh):
        if not isinstance(mass, np.ndarray):
            mass = np.array([mass])
            age = np.array([age])
            feh = np.array([feh])
        # Query the interpolator
        X_query = np.vstack([mass, age, feh]).T
        # Interpolate
        df = pd.DataFrame(
            self.l_interp(X_query),
            columns=self.predict_clean_names
        )
        return df


class ParsecBase(ICBase):
    """Handling PARSEC isochrones"""
    def __init__(self, dir_path, file_ending='dat'):
        super().__init__()
        # Save some PARSEC internal column names
        self.comment = r'#'
        self.colnames = {'header_start': '# Zini', 'teff': 'logTe'}
        self.post_process = {self.colnames['teff']: lambda x: 10 ** x}
        self.dir_path = dir_path
        self.flist_all = glob.glob(os.path.join(dir_path, f'*.{file_ending}'))

    def read_files(self, flist):
        frames = []
        for fname in flist:
            df_iso = self.read(fname)
            # Postprocessing
            for col, func in self.post_process.items():
                df_iso[col] = df_iso[col].apply(func)
            frames.append(df_iso)
        print('PARSEC isochrones read and processed!')
        df_final = pd.concat(frames)
        # Remove labels 8 & 9
        df_final = df_final[~df_final['label'].isin([7, 8, 9])]
        return df_final.sort_values(by=[self.colnames['age'], self.colnames['metal'], self.colnames['mass']])

    def read(self, fname):
        """
        Fetches the coordinates of a single isochrone from a given file and retrurns it
        :param fname: File name containing information on a single isochrone
        :return: x-y-Coordinates of a chosen isochrone, age, metallicity
        """
        # delim_whitespace is deprecated --> use sep='\s+' instead
        # df_iso = pd.read_csv(fname, delim_whitespace=True, comment=self.comment, header=None)
        df_iso = pd.read_csv(fname, sep='\s+', comment=self.comment, header=None)
        # Read first line and extract column names
        with open(fname) as f:
            for line in f:
                if line.startswith(self.colnames['header_start']):
                    break
        line = re.sub(r'\t', ' ', line)  # remove tabs
        line = re.sub(r'\n', ' ', line)  # remove newline at the end
        line = re.sub(self.comment, ' ', line)  # remove '#' at the beginning
        col_names = [elem for elem in line.split(' ') if elem != '']
        # Add column names
        df_iso.columns = col_names
        return df_iso


class ParsecGaia(ParsecBase):
    """Handling Gaia (E)DR3 photometric system"""

    def __init__(self, dir_path, file_ending='dat'):
        super().__init__(dir_path, file_ending)
        # Save some PARSEC internal column names
        self.colnames = {
            'mass': 'Mass',
            'logg': 'logg',
            'teff': 'logTe',
            'age': 'logAge',
            'metal': 'MH',
            'gmag': 'Gmag',
            'bp': 'G_BPmag',
            'rp': 'G_RPmag',
            'header_start': '# Zini'
        }
        # Save data and rename columns
        self.data = self.read_files(self.flist_all)
        # Prepare interpolation method
        self.cols_input = [self.colnames['mass'], self.colnames['age'], self.colnames['metal']]
        self.cols_predict = [
            self.colnames['gmag'], self.colnames['bp'], self.colnames['rp'],
            self.colnames['logg'], self.colnames['teff']
        ]
        self.predict_clean_names = ['M_G', 'G_BP', 'G_RP', 'logg', 'teff']
        self.fit_interpolator(n_skip=5)


class ParsecIR(ParsecBase):
    """Handling IR (2MASS, WISE, Spitzer) photometric system"""
    def __init__(self, dir_path, file_ending='dat'):
        super().__init__(dir_path, file_ending)
        # Save some PARSEC internal column names
        self.colnames = {
            'mass': 'Mass',
            'logg': 'logg',
            'teff': 'logTe',
            'age': 'logAge',
            'metal': 'MH',
            'Jmag': 'Jmag',
            'Hmag': 'Hmag',
            'Ksmag': 'Ksmag',
            'Irac1': 'IRAC_3.6mag',
            'Irac2': 'IRAC_4.5mag',
            'Irac3': 'IRAC_5.8mag',
            'Irac4': 'IRAC_8.0mag',
            'Mips1': 'MIPS_24mag',
            'mips2': 'MIPS_70mag',
            'mips3': 'MIPS_160mag',
            'W1': 'W1mag',
            'W2': 'W2mag',
            'W3': 'W3mag',
            'W4': 'W4mag',
            'header_start': '# Zini'
        }
        # Save data and rename columns
        self.data = self.read_files(self.flist_all)
        # Prepare interpolation method
        self.cols_input = [self.colnames['mass'], self.colnames['age'], self.colnames['metal']]
        self.cols_predict = [
            # 2MASS
            self.colnames['Jmag'], self.colnames['Hmag'], self.colnames['Ksmag'],
            # Spitzer
            self.colnames['Irac1'], self.colnames['Irac2'],
            self.colnames['Irac3'], self.colnames['Irac4'], self.colnames['Mips1'],
            # WISE
            self.colnames['W1'], self.colnames['W2'], self.colnames['W3'], self.colnames['W4'],
            # logg, teff
            self.colnames['logg'], self.colnames['teff']
        ]
        self.predict_clean_names = [
            'J', 'H', 'Ks', 'IRAC-1', 'IRAC-2', 'IRAC-3', 'IRAC-4', 'MIPS-1', 'W1', 'W2', 'W3', 'W4',
            'logg', 'teff'
        ]
        self.fit_interpolator(n_skip=5)
