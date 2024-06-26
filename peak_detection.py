import pandas as pd
import numpy as np
from scipy.signal import find_peaks, convolve, savgol_filter

class PeakDetection:
    """
    Class to process an .abf signal, computing baseline and floor signals, and detecting peaks in a signal
    """
    def __init__(self, data, time_column='time_index', signal_column='signal_value'):
        self.data = data
        self.time_column = time_column
        self.signal_column = signal_column
        self.smooth = self.smooth_signal()

    
    def compute_baseline(self, target="smooth", window_size=175, peak_prominence=0.2, smoothing_window_length=201, polyorder=3):
        """
        Function to compute the baseline signal
        :param window_size: int
        :param peak_prominence: float
        :param smoothing_window_length: int
        :param polyorder: int
        :return: list
        """
        if target == "smooth":
            y = np.array(self.smooth)
        elif target == "raw":
            y = np.array(self.data[self.signal_column])
        else:
            raise ValueError("target must be either 'smooth' or 'raw'")
        
        end_point = len(y) - window_size
        floor = np.empty([end_point])

        for i in range(end_point):
            frame = y[i:i + window_size]

            valleys, _ = find_peaks(-frame)

            if len(valleys) > 0:
                floor[i] = frame[valleys].min()
            else:
                floor[i] = np.nan  
                
        padding = smoothing_window_length // 2

        # Pad the resulting signal
        floor_padded = np.concatenate([floor[:padding], floor, floor[-padding:]]) # remove padding

        # Smooth the signal
        floor_smooth = savgol_filter(floor_padded, window_length=smoothing_window_length, polyorder=polyorder, mode='nearest')

        return floor_smooth.tolist()
    
    
    def compute_ceiling(self, target="smooth", window_size=175, peak_prominence=0.2, smoothing_window_length=201, polyorder=3):
        """
        Function to compute the ceiling signal
        :param window_size: int
        :param peak_prominence: float
        :param smoothing_window_length: int
        :param polyorder: i nt
        :return: list
        """
        if target == "smooth":
            y = np.array(self.smooth)
        elif target == "raw":
            y = np.array(self.data[self.signal_column])
        else:
            raise ValueError("target must be either 'smooth' or 'raw'")

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

        padding = smoothing_window_length // 2
        
        # Pad the resulting signal
        ceiling_padded = np.concatenate([ceiling[:padding], ceiling, ceiling[-padding:]])

        # Smooth the signal
        ceiling_smooth = savgol_filter(ceiling_padded, window_length=smoothing_window_length, polyorder=polyorder, mode='nearest')

        return ceiling_smooth.tolist()
    

    def smooth_signal(self, smoothing_window_length=51, polyorder=7):
        """
        Function to smooth the raw signal
        :param window_length: int
        :param polyorder: int
        :return: list
        """
        y = np.array(self.data[self.signal_column])

        # Pad the signal
        padding = smoothing_window_length // 2
        y_padded = np.concatenate([y[:padding], y, y[-padding:]])

        # Smooth the signal using a Savitzky-Golay filter
        smoothed_signal = savgol_filter(y_padded, window_length=smoothing_window_length, polyorder=polyorder, mode='nearest')

        # Remove the padding
        smoothed_signal = smoothed_signal[padding:-padding]

        return smoothed_signal.tolist()