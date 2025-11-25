
import numpy as np
import scipy.fft as fft
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt

def modify_rc():
    mpl.rcParams['figure.dpi'] = 250
    mpl.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'''\usepackage{bm}
\usepackage{xcolor}''')
    #mpl.rcParams['text.latex.preamble']=[r"\usepackage{bm}", r"\usepackage{xcolor}"]
    mpl.rc('font', family='serif', serif='Computer Modern', size=8)

# NOTE: COMMENT THIS OUT IF YOU DON'T HAVE LATEX CONFIGURED FOR MATPLOTLIB
modify_rc()
modify_rc()

def pretty_axes(axx):
    axx.yaxis.set_ticks_position('both')
    axx.xaxis.set_ticks_position('both')
    axx.tick_params(axis='y', direction='in')
    axx.tick_params(axis='y', direction='in', which='minor')
    axx.tick_params(axis='x', direction='in')
    axx.tick_params(axis='x', direction='in', which='minor')
    axx.grid(linestyle=':', alpha=0.3, linewidth=0.5, color='gray')

def get_minmax(ar, p=0, sym=True):
    if p != 0:
        cmin, cmax = np.nanpercentile(ar, p), np.nanpercentile(ar, 100-p)
    else:
        cmin, cmax = np.nanmin(ar), np.nanmax(ar)
    if sym:
        vmax = np.nanmax([np.abs(cmin), np.abs(cmax)])
        vmin = -vmax
    else:
        vmin, vmax = cmin, cmax
    return vmin, vmax

def calc_ilc(maps, a=None):
    """Calculate ILC
    Args:
        maps (list): List of maps for each frequency
        a (np.ndarray): Frequency-dependence
    Returns:
        ilc_map (np.ndarray): Corresponding ILC map
        w (np.ndarray): Weight values
    """
    T = np.array(maps)
    nfreq, npix = np.shape(T)
    C = np.cov(T)
    Cinv = np.linalg.inv(C)
    if a is None:
        a = np.ones(nfreq)
    w = Cinv @ a
    w = w / (np.transpose(a) @ Cinv @ a)
    ilc_map = w @ T
    return ilc_map, w

"""Define Dl Calculation Function"""

def calc_coherence(map1, map2):
    ClXY = hp.anafast(map1, map2)
    ClXX = hp.anafast(map1)
    ClYY = hp.anafast(map2)
    return np.abs(ClXY)/np.sqrt(ClXX*ClYY)

def calc_cl(map1, map2=None, mask=None, remove_dipole=False, pol=False, galcut=0., galcut_apo=None):
    """Calculate Cl

    Args:
        map1, map2: Maps for auto/cross spectra
        mask: Mask
        remove (bool): Remove the dipole
        pol (bool): If true, input [I,Q,U] maps
    Returns:
        ell, Dl
    """
    # gal_cut = (5.*u.degree).to(u.radian).value
    if galcut_apo is not None:
        cut_mask = hp.query_strip(hp.npix2nside(len(map1)), np.pi/2.-galcut, np.pi/2.+galcut)
        mask = np.ones_like(map1)
        mask[cut_mask] = 0
        mask = hp.smoothing(mask, fwhm=galcut_apo)
        # fsky_5 = np.sum(mask_5)/len(mask_5)
        # print(fsky_5)
        galcut = 0.

    if map2 is None:
        cl_mask = hp.ma(map1)
        if mask is not None:
            cl_mask.mask = np.logical_not(mask)
        if remove_dipole:
            cl_mask = hp.remove_dipole(cl_mask)
        print('Calculating spectrum')
        Cl = hp.anafast(cl_mask, pol=pol, gal_cut=galcut)
        ell = np.arange(np.shape(Cl)[-1])
    else:
        ## Cross-spectrum
        cl_mask1 = hp.ma(map1)
        cl_mask2 = hp.ma(map2)
        if mask is not None:
            cl_mask1.mask = np.logical_not(mask)
            cl_mask2.mask = np.logical_not(mask)
        if remove_dipole:
            cl_mask1 = hp.remove_dipole(cl_mask1)
            cl_mask2 = hp.remove_dipole(cl_mask2)
        print('Calculating spectrum')
        Cl = hp.anafast(cl_mask1, cl_mask2, pol=pol, gal_cut=galcut)
        ell = np.arange(np.shape(Cl)[-1])
    return ell, Cl

def calc_Dl(map1, map2=None, mask=None, remove_dipole=False, pol=False):
    ell, cl = calc_cl(map1, map2, mask, remove_dipole, pol)
    return ell, ell*(ell+1.)*cl/(2.*np.pi)

from astropy import wcs

def patch_creation(n_pix, delta_pix, pos_long, pos_lat):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.cdelt = np.array([-delta_pix, delta_pix])
    w.wcs.crval = [pos_long, pos_lat]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    patch = np.zeros((n_pix, n_pix))
    return (w, patch)

def fill_patch(w, patch, map_fill):
    patch_index = np.indices(np.shape(patch))
    lon, lat = w.wcs_pix2world(patch_index[1], patch_index[0], 0)
    get_pix_sky = hp.ang2pix(hp.get_nside(map_fill), lon, lat, lonlat=True)
    all_pix_values = map_fill[get_pix_sky]
    filled_patch = np.reshape(all_pix_values, np.shape(patch))
    return filled_patch

def w_object(n_pix, delta_pix, pos_long, pos_lat):
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [n_pix / 2, n_pix / 2]
    w.wcs.cdelt = np.array([-delta_pix, delta_pix])
    w.wcs.crval = [pos_long, pos_lat]
    w.wcs.ctype = ["GLON-TAN", "GLAT-TAN"]
    return w

def make_hist(map, log=False, range=None, bin_num=200, rescale=False, density=False):
    """Return histogram of observation at frequency f"""
    if rescale:
        map = map.copy() / (2.*np.nanstd(map))

    if range is None:
        range = (np.nanmin(map), np.nanmax(map))

    if log:
        bins = np.exp(np.linspace(np.log(range[0]), np.log(range[1]), bin_num))
        range = None
    else:
        bins = bin_num

    h, e = np.histogram(map, bins=bins, density=density, range=range)
    return h, e

def fullsky_k_scalefilter(maps, l_range, spin=2):
    """Select scale using the alms"""
    shape = np.shape(maps)
    if len(shape) == 1:
        ## Given just 1 map
        maps = [maps,]
        num_maps = 1
        npix = shape[0]
    else:
        num_maps = shape[0]
        npix = shape[1]

    if num_maps != 2 and spin == 2:
        raise ValueError('Wrong number of maps for spin 2')

    ## Calculate Alms
    if spin == 0:
        alms = hp.map2alm(maps)
    elif spin == 2:
        alms = hp.map2alm_spin(maps, spin=2)
    else:
        raise ValueError('Cannot do spin %s' % str(spin))

    nside = hp.npix2nside(npix)
    lmax = 3*nside - 1

    filter_min, filter_max = l_range

    # ## Scale filter (boxcar in alm-space)
    if len(np.shape(alms)) > 1:
        ## More than 1 map provided
        for j in range(np.shape(alms)[0]):
            for i in range(np.shape(alms)[1]):
                l, m = hp.Alm.getlm(lmax, i)
                if filter_min is None:
                    ## Low-pass filter
                    if l > filter_max:
                        alms[j][i] = 0. + 1j * 0.
                elif filter_max is None:
                    ## High-pass filter
                    if l < filter_min:
                        alms[j][i] = 0. + 1j * 0.
                else:
                    ## Band-pass filter
                    if l < filter_min or l > filter_max:
                        alms[j][i] = 0. + 1j * 0.
    else:
        ## 1 map provided
        for i in range(np.shape(alms)[0]):
            l, m = hp.Alm.getlm(lmax, i)
            if filter_min is None:
                ## Low-pass filter
                if l > filter_max:
                    alms[i] = 0. + 1j * 0.
            elif filter_max is None:
                ## High-pass filter
                if l < filter_min:
                    alms[i] = 0. + 1j * 0.
            else:
                ## Band-pass filter
                if l < filter_min or l > filter_max:
                    alms[i] = 0. + 1j * 0.

    ## Inverse transform to physical scale map
    if spin == 0:        
        scale_map = hp.alm2map(alms, nside=nside, lmax=lmax)
    elif spin == 2:
        scale_map = hp.alm2map_spin(alms, nside=nside, spin=2, lmax=lmax)
    return scale_map

def flat_k_scalefilter(map, krange):
    """fourier_scale(ar, krange)
    
    Selects frequencies/wavenumbers from `ar` within `krange`

    Args:
        ar (np.ndarray): Data to apply a Fourier filter to
        (kmin, kmax) (tuple): The minimum and maximum wavenumbers (in terms of k*L)
            if kmax is None:
                high-pass filter, i.e. keep everything >= kmin
            if kmin is None:
                low-pass filter, i.e. keep everything <= kmax
            else:
                band-pass filter, i.e. keep everything betweek kmin and kmax 
    Returns:
        np.ndarray: Real-space image containing frequencies within krange
    """
    # kmin, kmax = scale/1.25, scale*1.25
    kmin, kmax = krange
    uk2 = fft.fftshift(fft.fftn(map))
    dx = [2.*np.pi/N for N in map.shape]
    kvec = [fft.fftshift(fft.fftfreq(N))*2.*np.pi/dx[i] for i, N in enumerate(map.shape)]
    kmesh = np.meshgrid(*kvec, indexing='xy')
    k = np.linalg.norm(kmesh, axis=0)
    kmask = np.ones_like(k, dtype='bool')
    if kmin is None:
        ## Low-pass filter
        kmask[np.abs(k) <= kmax] = False
    elif kmax is None:
        ## High-pass filter
        kmask[np.abs(k) >= kmin] = False
    else:
        ## Band-pass filter
        kmask[np.logical_and(np.abs(k) <= kmax, np.abs(k) >= kmin)] = False
    uk2_filtered = uk2.copy()
    uk2_filtered[kmask] = 0. + 1j * 0.
    ar_filtered = fft.ifftn(fft.ifftshift(uk2_filtered)).real
    return ar_filtered

def fullsky_DoG_scalefilter(maps, scale, xi=1e-3):
    """Use Difference-of-Gaussians to select scale"""
    s1 = scale/np.sqrt(1. + xi)
    s2 = scale*np.sqrt(1. + xi)

    maps1 = hp.smoothing(maps, sigma=s1)
    maps2 = hp.smoothing(maps, sigma=s2)
    dist = maps1 - maps2

    return dist

def QU_to_EB(Q, U, nside):
    alm_E, alm_B = hp.map2alm_spin([Q, U], spin=2)
    map_E = hp.alm2map(alm_E, nside=nside)
    map_B = hp.alm2map(alm_B, nside=nside)
    return map_E, map_B

