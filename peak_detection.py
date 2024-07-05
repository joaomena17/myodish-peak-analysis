import pandas as pd
import numpy as np
import peakutils
import plotly.graph_objects as go
from scipy.signal import find_peaks, convolve, savgol_filter


class PeakDetection:
    """
    Class to process an .abf signal, computing baseline and floor signals, and detecting peaks in a signal
    """
    def __init__(self, data, time_column='time_index', signal_column='signal_value'):
        """
        data: pd.DataFrame
        time_column: str
        signal_column: str
        """
        self.data = data
        self.time_column = time_column
        self.signal_column = signal_column
        self.smooth = self.smooth_signal()

    
    def compute_baseline(self, target="smooth", window_size=350, peak_prominence=0.2, smoothing_window_length=301, polyorder=3):
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
                
        
        padding = max(window_size, smoothing_window_length) // 2

        # Pad the resulting signal
        floor_padded = np.concatenate([floor[:padding], floor, floor[-padding:]]) # remove padding

        # Smooth the signal
        floor_smooth = savgol_filter(floor_padded, window_length=smoothing_window_length, polyorder=polyorder, mode='nearest')

        return floor_smooth.tolist()
            
    
    def compute_ceiling(self, target="smooth", window_size=350, peak_prominence=0.2, smoothing_window_length=301, polyorder=3):
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

        padding = max(window_size, smoothing_window_length) // 2
        
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
    

    def diastolic_reference(self, ceiling=None, baseline=None, relative_to_baseline=0.1):
        """
        Function to compute the signal 10% higher than the diastolic force, used as a reference for peak detection
        This signal is used to compute time-to-peak (TTP) and time-to-relaxation (TTR)
        :param relative_to_baseline: float
        :return: list
        """
        if baseline is None:
            baseline = self.compute_baseline()
    
        if ceiling is None:
            ceiling = self.compute_ceiling()

        diastolic = [baseline[i] + relative_to_baseline * (ceiling[i] - baseline[i]) for i in range(len(baseline))]

        return diastolic
    

    def detect_peaks(self, rel_height=0.9, prominence=0.7, width=1, threshold=0.7, min_distance=50, threshold_absolute=False, wlen=250):
        """
        Function to detect peaks in the signal
        :param rel_height: float
        :param prominence: float
        :param width: int
        :param threshold: float
        :param min_distance: int
        :param threshold_absolute: bool
        :param wlen: int
        :return: list, list, list
        """

        peaks, _ = find_peaks(self.smooth, width=width, rel_height=rel_height, prominence=prominence, wlen=wlen)
        ceiling = self.compute_ceiling()
        baseline = self.compute_baseline()

        indices = []
        for i in range(len(peaks)):
            if self.smooth[peaks[i]] < ((ceiling[peaks[i]] - baseline[peaks[i]]) * threshold) + baseline[peaks[i]]:
               indices.append(i)

        peaks = np.delete(peaks, indices).tolist()
        peaks_timestamps = self.data[self.time_column][peaks].tolist()
        peaks_vals = np.array(self.smooth)[peaks].tolist()
                
        return peaks, peaks_timestamps, peaks_vals
    

    def plot(self, plot_all=False, plot_raw_signal=False, plot_smooth_signal=True, plot_peaks=True, plot_baseline=False, plot_ceiling=False, plot_diastolic=False,
             peaks=None, baseline=None, ceiling=None, diastolic=None):
        """
        Function to plot the signal with the detected peaks
        :param plot_all: bool
        :param plot_raw_signal: bool
        :param plot_smooth_signal: bool
        :param plot_peaks: bool
        :param plot_baseline: bool
        :param plot_ceiling: bool
        :param plot_diastolic: bool
        :param peaks: list
        :param baseline: list
        :param ceiling: list
        :param diastolic: list   
        """

        layout = go.Layout(title=f'Channel',
            xaxis=dict(
                visible=False, 
                autorange=True,
                fixedrange=False, 
                showgrid=False, 
                type="linear"
            ),
            yaxis=dict(
                visible=True, 
                autorange=True,
                fixedrange=True,
                showgrid=True
            ),
            dragmode='pan',
            annotations=[],
            hovermode='x unified'
        )

        
        fig = go.Figure(layout=layout)

        if plot_raw_signal or plot_all:

            fig.add_trace(go.Scatter
                (
                    x=self.data[self.time_column],
                    y=self.data[self.signal_column],
                    mode='lines',
                    name='Raw Signal',
                )
            )


        if plot_ceiling or plot_all:
            
            if ceiling is not None:
                fig.add_trace(go.Scatter
                    (
                        x=self.data[self.time_column],
                        y=ceiling,
                        mode='lines',
                        name='Ceiling',
                    )
                )

            else:
                fig.add_trace(go.Scatter
                    (
                        x=self.data[self.time_column],
                        y=self.compute_ceiling(),
                        mode='lines',
                        name='Ceiling',
                    )
                )


        if plot_baseline or plot_all:

            if baseline is not None:
                fig.add_trace(go.Scatter
                    (
                        x=self.data[self.time_column],
                        y=baseline,
                        mode='lines',
                        name='Baseline',
                    )
                )

            else:
                fig.add_trace(go.Scatter
                    (
                        x=self.data[self.time_column],
                        y=self.compute_baseline(),
                        mode='lines',
                        name='Baseline',
                    )
                )


        if plot_smooth_signal or plot_all:

            fig.add_trace(go.Scatter
                (
                    x=self.data[self.time_column],
                    y=self.smooth,
                    mode='lines',
                    name='Smooth Signal',
                )
            )
        

        if plot_diastolic or plot_all:

            if diastolic is not None:
                fig.add_trace(go.Scatter
                    (
                        x=self.data[self.time_column],
                        y=diastolic,
                        mode='lines',
                        name='Diastolic Reference',
                        line=dict(color='tomato', dash='dash')
                    )
                )

            else:    
                fig.add_trace(go.Scatter
                    (
                        x=self.data[self.time_column],
                        y=self.diastolic_reference(),
                        mode='lines',
                        name='Diastolic Reference',
                        line=dict(color='tomato', dash='dash')
                    )
                )


        if plot_peaks or plot_all:

            if peaks is not None:
                _, peak_idx, peak_vals = peaks
                fig.add_trace(go.Scatter
                    (
                        x=peak_idx,
                        y=peak_vals,
                        mode='markers',
                        name='Detected Peaks',
                        marker=dict(size=7)
                    )
                )

            else:    
                _, peak_idx, peak_vals = self.detect_peaks()
                fig.add_trace(go.Scatter
                    (
                        x=peak_idx,
                        y=peak_vals,
                        mode='markers',
                        name='Detected Peaks',
                        marker=dict(size=7)
                    )
                )


        fig.update_layout(
            title='Signal',
            xaxis_title='Time (s)',
            yaxis_title='Signal',
            showlegend=True
        )

        fig.show()
        