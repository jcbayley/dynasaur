import scipy
import numpy as np
from dynasaur.basis_functions import basis


def fit_cheby_to_hann(times, order=6, basis_type="chebyshev"):
    hwin = np.hanning(len(times))[np.newaxis, :]
    hann_cheb = basis[basis_type]["fit"](times, hwin, order)[0]
    return hann_cheb

def fit_cheby_to_tukey(times, alpha=0.5, order=6, basis_type="chebyshev"):
    hwin = scipy.signal.windows.tukey(len(times), alpha=alpha)[np.newaxis, :]
    tuk_cheb = basis[basis_type]["fit"](times, hwin, order)[0]
    return tuk_cheb


def chebint2(times, coeffs, basis_type="chebyshev", vel_pos_offsets=None):
    """compute the second integral correcting for offsets in the integrated values

    Args:
        times (_type_): _description_
        coeffs (_type_): _description_
        basis_type (string): type of basis used
        vel_pos_offsets (list): [position coefficients, velocity coefficients]

    Returns:
        _type_: _description_
    """
    if basis_type == "timeseries":
        if vel_pos_offsets is not None:
            pos_offset_co, vel_offset_co = vel_pos_offsets
            win_co_pos = basis[basis_type]["integrate"](coeffs, m=1, amps0=pos_offset_co, amps05=vel_offset_co)
        else:
            win_co_pos = basis[basis_type]["integrate"](coeffs, m=2)
    else:
        if vel_pos_offsets is not None:
            # get coefficients for position and velocity before windowing
            pos_offset_co, vel_offset_co = vel_pos_offsets
            win_co_vel = basis[basis_type]["integrate"](coeffs, m=1)
            # compute the values and subtract the mean from the first coefficient
            # this is so that there is not a velocity offset
            win_co_vel[np.isnan(win_co_vel)] = 0
            win_vel = basis[basis_type]["val"](times, win_co_vel)
            old_win_vel = basis[basis_type]["val"](times, vel_offset_co)
            # find difference in velocity at center of timeseries
            diff_vel = win_vel[int(0.5*len(win_vel))] - old_win_vel[int(0.5*len(old_win_vel))]
            win_co_vel[0] += diff_vel*len(win_vel)

            # now find the position
            win_co_pos = basis[basis_type]["integrate"](win_co_vel, m=1)
            # compute the values and subtract the mean from the first coefficient
            # this is so that there is not a position offset
            win_co_pos[np.isnan(win_co_pos)] = 0
            win_pos = basis[basis_type]["val"](times, win_co_pos)
            old_win_pos = basis[basis_type]["val"](times, pos_offset_co)
            # find difference between positions at center of timeseries
            diff_pos = win_pos[int(0.5*len(win_pos))] - old_win_pos[int(0.5*len(old_win_pos))]
            win_co_pos[0] += diff_pos*len(win_pos)
        else:
            win_co_pos = basis[basis_type]["integrate"](coeffs, m=2)
            win_co_pos[np.isnan(win_co_pos)] = 0

    return win_co_pos

def window_coeffs(times, coeffs, window_coeffs, basis_type="chebyshev", sub_mean=False):
    """window coefficients

    Args:
        times (_type_): array time timestamps
        coeffs (_type_): (n_coeffs, n_dimensions)
        window_coeffs (_type_): (n_coeffs)
        basis_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """

    subtract_offset = True
    if np.shape(coeffs)[1] != 3:
        raise Exception(f"Second dimensions of coeffs must be 3 dimensions not {np.shape(coeffs)[1]}")
    #hann_coeffs = np.array([ 3.47821791e-01,  1.52306260e-16, -4.85560481e-01, -5.11827799e-17, 1.51255010e-01,  2.65316279e-17, -1.48207898e-02])
    # find the acceleration components for each dimension
    co_x_acc = basis[basis_type]["derivative"](coeffs[:,0], m=2)
    co_y_acc = basis[basis_type]["derivative"](coeffs[:,1], m=2)
    co_z_acc = basis[basis_type]["derivative"](coeffs[:,2], m=2)

    if subtract_offset:
        co_x_vel = basis[basis_type]["derivative"](coeffs[:,0], m=1)
        co_y_vel = basis[basis_type]["derivative"](coeffs[:,1], m=1)
        co_z_vel = basis[basis_type]["derivative"](coeffs[:,2], m=1)

    # window each dimension in acceleration according to hann window
    win_co_x_acc = basis[basis_type]["multiply"](co_x_acc, window_coeffs)
    win_co_y_acc = basis[basis_type]["multiply"](co_y_acc, window_coeffs)
    win_co_z_acc = basis[basis_type]["multiply"](co_z_acc, window_coeffs)

    # fix bug when object not moving in z axes (repeat 0 for n coeffs)
    if len(win_co_x_acc) == 1:
        win_co_x_acc = np.repeat(win_co_x_acc[0], len(win_co_y_acc))
    if len(win_co_y_acc) == 1:
        win_co_y_acc = np.repeat(win_co_y_acc[0], len(win_co_z_acc))
    if len(win_co_z_acc) == 1:
        win_co_z_acc = np.repeat(win_co_z_acc[0], len(win_co_y_acc))
    # integrate the windowed acceleration twice to get position back
    #win_co_x = basis[basis_type]["integrate"](win_co_x_acc, m=2)
    #win_co_y = basis[basis_type]["integrate"](win_co_y_acc, m=2)
    #win_co_z = basis[basis_type]["integrate"](win_co_z_acc, m=2)

    win_co_x = chebint2(
        times, 
        win_co_x_acc, 
        basis_type=basis_type,
        vel_pos_offsets=[coeffs[:,0], co_x_vel])
    win_co_y = chebint2(
        times, 
        win_co_y_acc, 
        basis_type=basis_type,
        vel_pos_offsets=[coeffs[:,1], co_y_vel])
    win_co_z = chebint2(
        times, 
        win_co_z_acc, 
        basis_type=basis_type,
        vel_pos_offsets=[coeffs[:,2], co_z_vel])
    
    coarr = np.array([win_co_x, win_co_y, win_co_z]).T
    return coarr

def get_window_coeffs(times, window, order=6, basis_type="fourier", alpha=0.5):

    if basis_type == "fourier":
        order = int(order/2) + 1

    if window == "tukey":
        win_coeffs = fit_cheby_to_tukey(times, alpha=alpha, order=order, basis_type=basis_type)
    elif window == "hann":
        win_coeffs = fit_cheby_to_hann(times, order=order, basis_type=basis_type)
    else:
        raise Exception(f"Window {window} does not Exist")
    
    return win_coeffs

def perform_window(times, coeffs, window, order=6, basis_type="chebyshev", sub_mean=False, alpha=0.5):
    """_summary_

    Args:
        times (_type_):  n_times
        coeffs (_type_):  (n_coeffs, n_dimensions)
        window (_type_): (tukey or hann)
    """

    #if basis_type == "fourier":
    #    order = int(order/2) + 1
        
    if window != "none":
        win_coeffs = get_window_coeffs(times, window, order=order, basis_type=basis_type, alpha=alpha)
    
        coeffs = window_coeffs(times, coeffs, win_coeffs, basis_type=basis_type, sub_mean=sub_mean)
    else:
        win_coeffs = None

    return coeffs, win_coeffs