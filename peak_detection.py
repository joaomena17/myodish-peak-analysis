import pandas as pd
import numpy as np
from scipy.signal import find_peaks, convolve

class PeakDetection:
    """
    Class to process an .abf signal, computing baseline and floor signals, and detecting peaks in a signal
    """
    def __init__(self, data, time_column='time_index', signal_column='signal_value'):
        self.data = data
        self.time_column = time_column
        self.signal_column = signal_column


    
    def compute_baseline(self, window_size=100):
        """
        Function to compute the baseline signal
        :param window_size: int
        :return: pandas.Series
        """
        n_lowest = 25
        smoothing_window_size = np.max([350, 25])
        y = np.array(self.data[self.signal_column])
        end_point = len(y) - window_size
        floor = np.empty([end_point])

        for i in range(end_point):
            frame = y[i:i + window_size]

            # Find the valleys (local minima) in the frame by finding peaks in the inverted signal
            valleys, _ = find_peaks(-frame)

            # If there are any valleys, use the lowest valley as the floor for this window
            if len(valleys) > 0:
                floor[i] = frame[valleys].min()
            else:
                floor[i] = np.nan  # No floor defined if there are no valleys

        # add padding to prepare for the convolution
        padding = smoothing_window_size
        floor_padded = np.pad(floor, (padding, padding), 'edge')
        floor_padded = np.nan_to_num(floor_padded)

        # Smooth the floor using a moving average
        window = np.ones(smoothing_window_size) / smoothing_window_size
        floor_smooth = convolve(floor_padded, window, mode='valid', method='direct')

        return floor_smooth.tolist()
    
    def compute_ceiling(self, window_size=100):
        smoothing_window_size = np.max([350, 25])

        y = np.array(self.data[self.signal_column])
        end_point = len(y) - window_size
        ceiling = np.empty([end_point])

        for i in range(end_point):
            frame = y[i:i+ window_size]
            
            # Find the peaks in the frame
            peaks, _ = find_peaks(frame)
            
            # If there are any peaks, use the highest peak as the ceiling for this window
            if len(peaks) > 0:
                ceiling[i] = frame[peaks].max()
            else:
                ceiling[i] = np.nan  # No ceiling defined if there are no peaks

        padding = smoothing_window_size
        ceiling_padded = np.pad(ceiling, (padding, padding), 'edge')

        ceiling_padded = np.nan_to_num(ceiling_padded)

        # Smooth the ceiling using a moving average
        window = np.ones(smoothing_window_size)/smoothing_window_size
        ceiling_smooth = convolve(ceiling_padded, window, mode='valid', method='direct')

        return ceiling_smooth.tolist()