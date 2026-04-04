# -*- coding: utf-8 -*-
"""
Utility functions for D3 data reduction module.
Extracted from D3.py to improve code organization and reusability.

Created: 2026-04-04
Author: Refactored by Claude Code
"""

import numpy as np
import re
from datetime import datetime


def zero_divide(a, b):
    """
    Safely divide arrays, returning zero where division by zero would occur.

    Parameters
    ----------
    a : array_like
        Numerator
    b : array_like
        Denominator

    Returns
    -------
    array_like
        Result of a/b with zeros where b==0
    """
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)


def denan(arr):
    """
    Replace NaN and None values with zeros in 1D array.

    Parameters
    ----------
    arr : array_like
        Input array

    Returns
    -------
    array_like
        Array with NaN/None replaced by zeros
    """
    result = arr.copy()
    for i in range(len(result)):
        if result[i] is None or (isinstance(result[i], float) and np.isnan(result[i])):
            result[i] = 0
    return result


def denan_2d(data):
    """
    Replace NaN and None values with zeros in 2D array.

    Parameters
    ----------
    data : array_like
        Input 2D array

    Returns
    -------
    array_like
        Array with NaN/None replaced by zeros
    """
    result = data.copy()
    for i in range(len(result)):
        for j in range(len(result[0])):
            if result[i][j] is None or (isinstance(result[i][j], float) and np.isnan(result[i][j])):
                result[i][j] = 0
    return result


def time_diff(time1, time2):
    """
    Calculate time difference between two byte-format timestamps.

    Parameters
    ----------
    time1 : bytes
        Start time in format b'YYYY-MM-DD HH:MM:SS'
    time2 : bytes
        End time in format b'YYYY-MM-DD HH:MM:SS'

    Returns
    -------
    tuple
        (difference_seconds, difference_minutes)
    """
    time1_str = time1.decode('utf-8')
    time2_str = time2.decode('utf-8')
    time_format = '%Y-%m-%d %H:%M:%S'

    time1_obj = datetime.strptime(time1_str, time_format)
    time2_obj = datetime.strptime(time2_str, time_format)

    time_difference = time2_obj - time1_obj
    difference_seconds = time_difference.total_seconds()
    difference_minutes = difference_seconds / 60

    return difference_seconds, difference_minutes


def get_proton_charge_from_xml(file):
    """
    Extract proton charge from XML file (legacy method).

    Parameters
    ----------
    file : str
        Path to XML file

    Returns
    -------
    float
        Proton charge value
    """
    with open(file, 'r') as f:
        txt = f.readlines()
    pattern = re.compile(r'(\d+\.?\d*)</proton_charge>?')
    tmp = pattern.findall(str(txt))
    ProtonCharge = float(tmp[0])
    return ProtonCharge


def falling_distance(wavelength, L_1, L_2):
    """
    Calculate neutron falling distance due to gravity.

    Formula source: Bouleam SANS Tool Box: Chapter 17 - GRAVITY CORRECTING PRISMS

    Parameters
    ----------
    wavelength : float or array_like
        Neutron wavelength (Angstroms)
    L_1 : float
        Primary flight path (mm)
    L_2 : float
        Secondary flight path (mm)

    Returns
    -------
    float or array_like
        Falling distance (mm)
    """
    B = 3.073E-9 * 100
    L = L_1 + L_2
    y = B * wavelength**2 * L * (L_1 - L)
    y = y / 1000  # Convert to mm
    return y


def get_run_fold(run_num):
    """
    Format run number as 'RUN000XXXX' directory name.

    Parameters
    ----------
    run_num : str or int
        Run number

    Returns
    -------
    str
        Formatted run folder name
    """
    if isinstance(run_num, str):
        run_num_str = run_num.split('_')[0]
    else:
        run_num_str = str(run_num)

    run_fold = r"RUN" + str('0' * (7 - len(run_num_str))) + run_num_str
    return run_fold


def gaussian(x, A, sigma, mu):
    """
    Gaussian function.

    Parameters
    ----------
    x : array_like
        Independent variable
    A : float
        Amplitude
    sigma : float
        Standard deviation
    mu : float
        Mean

    Returns
    -------
    array_like
        Gaussian function values
    """
    return A * np.exp(-(x - mu)**2 / sigma**2)


def gaussian_fit(xx, yy):
    """
    Fit Gaussian peak to 1D data.

    Parameters
    ----------
    xx : array_like
        X coordinates
    yy : array_like
        Y values (intensities)

    Returns
    -------
    tuple
        (A, sigma, mu) - amplitude, width, and position
    """
    import scipy.optimize

    sigma_guess = (xx[-1] - xx[0])
    mu_guess = np.average(xx)
    aa, bb = scipy.optimize.curve_fit(
        lambda x, A, sigma, mu: gaussian(x, A, sigma, mu),
        xx, yy, p0=[3000, sigma_guess, mu_guess]
    )
    return aa[0], aa[1], aa[2]


def find_peaks(arr, peak_width):
    """
    Find peak positions in 1D array based on peak width.

    Parameters
    ----------
    arr : array_like
        1D data array
    peak_width : int
        Expected peak width in array indices

    Returns
    -------
    list
        List of tuples (peak_position, peak_intensity)
    """
    sum_match = []
    peak_pos2 = []

    for match in np.arange(len(arr) - peak_width):
        upper = match + peak_width
        summ = np.sum(arr[match:upper])
        sum_match.append((match, summ))

    for item in sum_match[peak_width:-peak_width]:
        if sum_match[item[0]][1] > sum_match[item[0] - 1][1] and \
           sum_match[item[0]][1] > sum_match[item[0] + 1][1]:
            up_bound = item[0] + peak_width
            matched_pos = np.array(sum_match[item[0]:up_bound])
            ave_pos = np.average(matched_pos[:, 0])
            peak_pos2.append((ave_pos, item[1]))

    return peak_pos2


def get_big_peaks(arr, peak_pos, num, peak_width):
    """
    Get refined positions of the strongest peaks using Gaussian fitting.

    Parameters
    ----------
    arr : array_like
        1D data array
    peak_pos : list
        List of (position, intensity) tuples from find_peaks
    num : int
        Number of strongest peaks to return
    peak_width : int
        Peak width for fitting window

    Returns
    -------
    array_like
        2D array of shape (num, 2) containing (position, intensity) for strongest peaks
    """
    peak_height_sorted = sorted(peak_pos, key=lambda x: x[1], reverse=True)
    big_peaks_sorted = sorted(peak_height_sorted[:num], key=lambda x: x[0])
    big_peaks_array = np.array(big_peaks_sorted)

    mu_array = []
    for i, peaks in enumerate(big_peaks_array[:, 0]):
        low_bound = int(peaks - peak_width / 2)
        high_bound = int(peaks + peak_width / 2)
        xx = np.arange(low_bound, high_bound)
        yy = arr[low_bound:high_bound]
        A, sigma, mu = gaussian_fit(xx, yy)
        big_peaks_array[i, 0] = mu
        mu_array.append(mu)

    out3 = np.ones(big_peaks_array.shape)
    out3[:, 0] = mu_array
    out3[:, 1] = big_peaks_array[:, 1]

    return out3
