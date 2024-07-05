import numpy as np
import pandas as pd
from peak_detection import PeakDetection

class PeakAnalysis:
    def __init__(self, data: PeakDetection):
        self.data = data
        self.peak_objects = []
        self.ttp_points = []
        self.ttr_points = []
        self.compute_attributtes()

    def compute_attributtes(self):
        peaks_idx, peaks_timestamp, peaks_val = self.data.detect_peaks()

        peaks =  zip(peaks_idx, zip(peaks_timestamp, peaks_val))
    
        baseline = self.data.compute_baseline()
        diastolic_reference_at_peaks = self.data.diastolic_reference(baseline=baseline)

        for (peak_idx, (peak_timestamp, peak_val)) in peaks:
            metadata = {}

            metadata['peak_idx'] = peak_idx
            metadata['peak_value'] = peak_val
            metadata['timestamp'] = peak_timestamp

            metadata['baseline_value'] = baseline[peak_idx]
            metadata['force'] = peak_val - baseline[peak_idx]

            # Calculate time to peak and time to relaxation
            i = 0

            if peak_idx == 0 or peak_idx >= len(self.data.smooth) - 2 or peak_idx >= len(diastolic_reference_at_peaks) - 2:
                metadata['time_to_peak'] = np.nan
                metadata['time_to_relaxation'] = np.nan
                metadata['duration'] = np.nan
                self.peak_objects.append(metadata)
                continue

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
                metadata['time_to_peak'] = (i)/100 # seconds
                self.ttp_points.append((peak_timestamp - i/100, self.data.smooth[peak_idx - i]))

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
                metadata['time_to_relaxation'] = (i)/100 # seconds
                self.ttr_points.append((peak_timestamp + i/100, self.data.smooth[peak_idx + i]))
    

            # Calculate total duration of peak
            if not np.isnan(metadata['time_to_peak']) and not np.isnan(metadata['time_to_relaxation']):
                metadata['duration'] = metadata['time_to_relaxation'] + metadata['time_to_peak']
            else:
                metadata['duration'] = np.nan

            self.peak_objects.append(metadata)



            