import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

frame_length = 64

hop = 16

fs = 48000

# Finds the onset by checking when the signal goes above a fixed threshold
def find_onset_peak_based(data, threshold_frac=0.1):
    envelope = np.abs(data)
    peak = np.max(envelope)
    onset = np.argmax(envelope > peak * threshold_frac)
    return onset

# Returns the onset 
def mic_onset_position(data):
    onset = find_onset_peak_based(data)
    
    return onset

# Windows a signal from the onset position to the window length
def window_signal(data, onset, window_length):
    windowed = data[onset : onset + window_length]
    
    return windowed