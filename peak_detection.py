import pandas as pd
import numpy as np
import peakutils
import plotly.graph_objects as go
from scipy.signal import find_peaks, convolve, savgol_filter


class PeakDetection:
    """
    Class to process an .abf signal, computing baseline and floor signals, and detecting peaks in a signal
    """
    def __init__(self, data, stim=None,
                 time_column='time_index', signal_column='signal_value',
                 smooth_sig_smoothing_window=51, smooth_sig_polyorder=7, # params for smooth signal
                 compute_from="smooth", aux_window_size=350, aux_peak_prominence=0.2, aux_smoothing_window=301, aux_polyorder=3, # params for ceiling and baseline functions
                 relative_to_baseline=0.1, # params for diastolic reference
                 rel_height=0.3, prominence=0.3, width=1, threshold=0.7, min_distance=20, wlen=250 # params for detect_peaks
                ):
        """
        data: pd.DataFrame
        time_column: str
        signal_column: str
        """
        self.data = data
        self.stim = stim

        # normalize values in stim to data values scale
        if self.stim is not None:
            self.stim[signal_column] = self.stim[signal_column] * (self.data[signal_column].max() / self.stim[signal_column].max())

        self.time_column = time_column
        self.signal_column = signal_column
        
        self.compute_from = compute_from
        self.smooth_sig_smoothing_window = smooth_sig_smoothing_window
        self.smooth_sig_polyorder = smooth_sig_polyorder
        
        self.aux_window_size = aux_window_size
        self.aux_peak_prominence = aux_peak_prominence
        self.aux_smoothing_window = aux_smoothing_window
        self.aux_polyorder = aux_polyorder
        
        self.relative_to_baseline = relative_to_baseline
        
        self.rel_height = rel_height
        self.prominence = prominence
        self.width = width
        self.threshold = threshold
        self.min_distance = min_distance
        self.wlen = wlen

        if self.compute_from == "smooth":
            self.smooth = self.smooth_signal()

    
    def compute_baseline(self, target=None, window_size=None, peak_prominence=None, smoothing_window_length=None, polyorder=None):
        """
        Function to compute the baseline signal
        :param window_size: int
        :param peak_prominence: float
        :param smoothing_window_length: int
        :param polyorder: int
        :return: list
        """
        if target is None:
            target = self.compute_from

        if window_size is None:
            window_size = self.aux_window_size
        
        if peak_prominence is None:
            peak_prominence = self.aux_peak_prominence

        if smoothing_window_length is None:
            smoothing_window_length = self.aux_smoothing_window

        if polyorder is None:
            polyorder = self.aux_polyorder


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
            
    
    def compute_ceiling(self, target=None, window_size=None, peak_prominence=None, smoothing_window_length=None, polyorder=None):
        """
        Function to compute the ceiling signal
        :param window_size: int
        :param peak_prominence: float
        :param smoothing_window_length: int
        :param polyorder: i nt
        :return: list
        """
        if target is None:
            target = self.compute_from

        if window_size is None:
            window_size = self.aux_window_size
        
        if peak_prominence is None:
            peak_prominence = self.aux_peak_prominence

        if smoothing_window_length is None:
            smoothing_window_length = self.aux_smoothing_window

        if polyorder is None:
            polyorder = self.aux_polyorder


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
    

    def smooth_signal(self, smoothing_window_length=None, polyorder=None):
        """
        Function to smooth the raw signal
        :param window_length: int
        :param polyorder: int
        :return: list
        """
        if smoothing_window_length is None:
            smoothing_window_length = self.smooth_sig_smoothing_window

        if polyorder is None:
            polyorder = self.smooth_sig_polyorder

        y = np.array(self.data[self.signal_column])

        # Pad the signal
        padding = smoothing_window_length // 2
        y_padded = np.concatenate([y[:padding], y, y[-padding:]])

        # Smooth the signal using a Savitzky-Golay filter
        smoothed_signal = savgol_filter(y_padded, window_length=smoothing_window_length, polyorder=polyorder, mode='nearest')

        # Remove the padding
        smoothed_signal = smoothed_signal[padding:-padding]

        return smoothed_signal.tolist()
    

    def diastolic_reference(self, ceiling=None, baseline=None, relative_to_baseline=None):
        """
        Function to compute the signal 10% higher than the diastolic force, used as a reference for peak detection
        This signal is used to compute time-to-peak (TTP) and time-to-relaxation (TTR)
        :param relative_to_baseline: float
        :return: list
        """
        if relative_to_baseline is None:
            relative_to_baseline = self.relative_to_baseline

        if baseline is None:
            baseline = self.compute_baseline()
    
        if ceiling is None:
            ceiling = self.compute_ceiling()

        diastolic = [baseline[i] + relative_to_baseline * (ceiling[i] - baseline[i]) for i in range(len(baseline))]

        return diastolic
    

    def detect_peaks(self, rel_height=None, prominence=None, width=None, threshold=None, min_distance=None, wlen=None):
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
        if rel_height is None:
            rel_height = self.rel_height

        if prominence is None:
            prominence = self.prominence

        if width is None:
            width = self.width

        if threshold is None:
            threshold = self.threshold

        if min_distance is None:
            min_distance = self.min_distance

        if wlen is None:
            wlen = self.wlen

        peaks, _ = find_peaks(self.smooth, width=width, distance=min_distance, rel_height=rel_height, prominence=prominence, wlen=wlen)
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
    

    def detect_stim_peaks(self, rel_height=None, prominence=None, width=None, threshold=None, min_distance=None, wlen=None):
        """
        Function to detect peaks in the stimulus signal
        :param rel_height: float
        :param prominence: float
        :param width: int
        :param threshold: float
        :param min_distance: int
        :param wlen: int
        :return: list
        """
        if rel_height is None:
            rel_height = self.rel_height

        if prominence is None:
            prominence = self.prominence

        if width is None:
            width = self.width

        if threshold is None:
            threshold = self.threshold

        if min_distance is None:
            min_distance = self.min_distance

        if wlen is None:
            wlen = self.wlen

        peaks, _ = find_peaks(self.stim[self.signal_column])

        return peaks
    

    def plot(self, plot_all=False, plot_raw_signal=False, plot_smooth_signal=True, plot_stim=False, plot_peaks=True, plot_baseline=False, plot_ceiling=False, plot_diastolic=False,
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


        if plot_stim or plot_all:
            
                if self.stim is not None:
                    stim_peaks = self.stim[self.time_column][self.detect_stim_peaks()]

                    for peak in stim_peaks:
                        fig.add_vline(x=peak, line=dict(color="LimeGreen", width=1, dash="dash"))

                    fig.add_annotation(
                        x=sum(stim_peaks) / len(stim_peaks),
                        y=max(self.data["signal_value"]) + 20,
                        text="Stimulus Reference",
                        showarrow=False,
                        font=dict(size=14, color="black"),
                        align="center"
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
        