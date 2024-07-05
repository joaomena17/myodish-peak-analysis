import numpy as np
import pandas as pd
import plotly.graph_objects as go
from peak_detection import PeakDetection

class PeakAnalysis:
    """
    Class to analyze peaks detected by PeakDetection class
    """
    def __init__(self, data: PeakDetection):
        """
        data: PeakDetection object
        """
        self.data = data
        self.peak_objects = []
        self.ttp_points = []
        self.ttr_points = []

        self.avg_ttp = np.nan
        self.avg_ttr = np.nan
        self.avg_duration = np.nan
        self.avg_force = np.nan    

        self.layout = go.Layout(title=f'Channel',
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
        
        self.compute_attributtes()
        self.compute_avg_attributtes()

    def compute_attributtes(self):
        """
        Compute attributes of the peaks
        """

        # extract data from the PeakDetection object
        peaks_idx, peaks_timestamp, peaks_val = self.data.detect_peaks()

        peaks =  zip(peaks_idx, zip(peaks_timestamp, peaks_val))
    
        baseline = self.data.compute_baseline()
        diastolic_reference_at_peaks = self.data.diastolic_reference(baseline=baseline)

        # traverse all the peaks
        for (peak_idx, (peak_timestamp, peak_val)) in peaks:
            metadata = {}

            metadata['peak_idx'] = peak_idx
            metadata['peak_value'] = peak_val
            metadata['timestamp'] = peak_timestamp

            metadata['baseline_value'] = baseline[peak_idx]
            metadata['force'] = peak_val - baseline[peak_idx]

            # Check if peak is at the edge of the signal
            if peak_idx == 0 or peak_idx >= len(self.data.smooth) - 2 or peak_idx >= len(diastolic_reference_at_peaks) - 2:
                metadata['time_to_peak'] = np.nan
                metadata['time_to_relaxation'] = np.nan
                metadata['duration'] = np.nan
                self.peak_objects.append(metadata)
                continue

            
            # Calculate time to peak
            i = 0
            ttp_edge_reference: bool = self.data.smooth[peak_idx - i] >= diastolic_reference_at_peaks[peak_idx - i]
            ttp_new_edge_reference =  not ttp_edge_reference

            while ttp_edge_reference != ttp_new_edge_reference and peak_idx - i > 0:
                # update i
                i+=1

                # update reference
                ttp_edge_reference = self.data.smooth[peak_idx - i] >= diastolic_reference_at_peaks[peak_idx - i]

            if i == 0:
                metadata['time_to_peak'] = np.nan
            else:
                metadata['time_to_peak'] = i/100 # seconds
                self.ttp_points.append((peak_timestamp - i/100, self.data.smooth[peak_idx - i]))

            # Calculate time to relaxation
            i = 0
            ttr_edge_reference = self.data.smooth[peak_idx + i] <= diastolic_reference_at_peaks[peak_idx + i]
            ttr_new_edge_reference =  not ttr_edge_reference

            while ttr_edge_reference != ttr_new_edge_reference and peak_idx + i < len(self.data.smooth) - 1:
                # update i
                i+=1

                # update reference
                ttr_edge_reference = self.data.smooth[peak_idx + i] <= diastolic_reference_at_peaks[peak_idx + i]
            
            if peak_idx + i == len(self.data.smooth) - 1:
                metadata['time_to_relaxation'] = np.nan
            else:
                metadata['time_to_relaxation'] = i/100 # seconds
                self.ttr_points.append((peak_timestamp + i/100, self.data.smooth[peak_idx + i]))    

            # Calculate total duration of peak
            if not np.isnan(metadata['time_to_peak']) and not np.isnan(metadata['time_to_relaxation']):
                metadata['duration'] = metadata['time_to_relaxation'] + metadata['time_to_peak']
            else:
                metadata['duration'] = np.nan

            self.peak_objects.append(metadata)


    def compute_avg_attributtes(self):
        """
        Compute average attributes of the peaks
        """
        ttp = [x['time_to_peak'] for x in self.peak_objects if not np.isnan(x['time_to_peak'])]
        ttr = [x['time_to_relaxation'] for x in self.peak_objects if not np.isnan(x['time_to_relaxation'])]
        duration = [x['duration'] for x in self.peak_objects if not np.isnan(x['duration'])]
        force = [x['force'] for x in self.peak_objects if not np.isnan(x['force'])]

        self.avg_ttp = np.mean(ttp)
        self.avg_ttr = np.mean(ttr)
        self.avg_duration = np.mean(duration)
        self.avg_force = np.mean(force)


    def plot_analysis_visualizer(self):
        """
        Plot the analysis of the peaks
        """
                
        ttr_x = [x[0] for x in self.ttr_points]
        ttr_y = [x[1] for x in self.ttr_points]
        ttp_x = [x[0] for x in self.ttp_points]
        ttp_y = [x[1] for x in self.ttp_points]

        timestamps = [x['timestamp'] for x in self.peak_objects]
        peaks = [x['peak_value'] for x in self.peak_objects]

        fig = go.Figure(layout=self.layout)

        fig.add_trace(go.Scatter(x=self.data.data[self.data.time_column], y=self.data.smooth, mode="lines", name="Signal"))
        fig.add_trace(go.Scatter(x=self.data.data[self.data.time_column], y=self.data.diastolic_reference(), mode="lines", name="10% Diastolic Reference", line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=timestamps, y=peaks, mode="markers", name="Peaks"))
        fig.add_trace(go.Scatter(x=ttp_x, y=ttp_y, mode="markers", name="TTP"))
        fig.add_trace(go.Scatter(x=ttr_x, y=ttr_y, mode="markers", name="TTR"))

        fig.show()


    def plot_force_evolution(self):
        """
        Plot the force evolution of the peaks
        """
        timestamps = [x['timestamp'] for x in self.peak_objects]
        forces = [x['force'] for x in self.peak_objects]

        fig = go.Figure(layout=self.layout)

        fig.add_trace(go.Scatter(x=timestamps, y=forces, mode="lines", name="Force"))

        fig.show()


    def plot_beat_times_evolution(self, all=False, ttp=False, ttr=False, duration=False):
        """
        Plot the evolution of the time attributes in the beats
        """
        timestamps = [x['timestamp'] for x in self.peak_objects]
        ttp = [x['time_to_peak'] for x in self.peak_objects]
        ttr = [x['time_to_relaxation'] for x in self.peak_objects]
        duration = [x['duration'] for x in self.peak_objects]

        fig = go.Figure(layout=self.layout)

        if ttp or all:
            fig.add_trace(go.Scatter(x=timestamps, y=ttp, mode="lines", name="TTP"))

        if ttr or all:
            fig.add_trace(go.Scatter(x=timestamps, y=ttr, mode="lines", name="TTR"))

        if duration or all:
            fig.add_trace(go.Scatter(x=timestamps, y=duration, mode="lines", name="Duration"))

        fig.show()

            