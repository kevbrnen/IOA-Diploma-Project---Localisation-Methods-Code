import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path    
from scipy.fft import fft, ifft
from scipy.optimize import brentq

def fft_cross_correlation(x, y):
    X = fft(x)
    Y = fft(y)
    corr = ifft(X * np.conj(Y))
    return np.fft.fftshift(np.real(corr))

def woodworth_itd(angle_deg, r, c=343.0):
    theta = np.radians(angle_deg)
    if abs(theta) <= np.pi/2:
        return (r / c) * (np.sin(theta) + theta)
    else:
        return (r / c) * (np.sign(theta) * np.pi - theta + np.sin(theta))

def itd_to_angle(itd_s, r, fs, c=343.0):
    max_itd = ((r / c) * (np.pi / 2 + 1) * fs * 1.5)
    itd_s = np.clip(itd_s, -max_itd, max_itd)
    f = lambda theta: (r/c) * (np.sin(theta) + theta) - itd_s
    try:
        return np.degrees(brentq(f, -np.pi/2 + 1e-6, np.pi/2 - 1e-6))
    except ValueError:
        return np.sign(itd_s) * 90.0
    
def xcorr_fft(L, R):
    n = len(L) + len(R) - 1
    if n % 2 == 0:
        n += 1
    L_fft = np.fft.rfft(L, n=n)
    R_fft = np.fft.rfft(R, n=n)
    return np.fft.irfft(L_fft * np.conj(R_fft))

def corr_to_itd(correlation, d, fs, c=343):
    half = len(correlation) // 2
    lags = np.arange(-half, half + 1)[:len(correlation)]
        
    max_itd_samples = int((d / c) * fs)
    mask = (lags >= -max_itd_samples) & (lags <= max_itd_samples)
    itd_samples = lags[mask][np.argmax(correlation[mask])]
    itd_seconds = itd_samples / fs
    
    return itd_seconds

def gcc_phat(x, y):
    if x.ndim > 1:
        x = x.mean(axis=1)
    if y.ndim > 1:
        y = y.mean(axis=1)

    n = len(x) + len(y) - 1
    n = 1 << (n - 1).bit_length()

    X = np.fft.rfft(x, n=n)

    Y = np.fft.rfft(y, n=n)

    cross = X * np.conj(Y)
    cross /= (np.abs(cross) + 1e-12)
    result = np.fft.irfft(cross)
    return np.fft.fftshift(result)