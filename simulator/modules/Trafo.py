import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import ICRS, Galactic


class Trafo:
    def __init__(self):
        pass

    @staticmethod
    def spher2cart(data):
        ra, dec, parallax, pmra, pmdec, rv = data.T
        dist = 1000 / parallax
        dist[dist < 0] = np.nan
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

    @staticmethod
    def cart2spher(data):
        X, Y, Z, U, V, W = data.T
        c = Galactic(
            u=X * u.pc, v=Y * u.pc, w=Z * u.pc,
            U=U * u.km / u.s, V=V * u.km / u.s, W=W * u.km / u.s,
            representation_type="cartesian",
            # Velocity representation
            differential_type="cartesian",
        )
        c_icrs = c.transform_to(ICRS())
        c_icrs.representation_type = 'spherical'
        # Get observables in ICRS
        ra, dec, dist = c_icrs.ra.value, c_icrs.dec.value, c_icrs.distance.value
        cos_dec = np.cos(np.radians(dec))
        pmra, pmdec, rv = c_icrs.pm_ra.value * cos_dec, c_icrs.pm_dec.value, c_icrs.radial_velocity.value
        parallax = 1000 / dist
        X = np.vstack([ra, dec, parallax, pmra, pmdec, rv]).T
        df = pd.DataFrame(X, columns=['ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity'])
        return df