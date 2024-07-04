import numpy as np
import pandas as pd
from peak_detection import PeakDetection

class PeakAnalysis:
    def __init__(self, data: PeakDetection):
        self.data = data
        self.peak_objects = []
        self.compute_forces()

    def compute_forces(self):
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
            i = peak_idx

            while self.data.smooth[i] != diastolic_reference_at_peaks[i] and i > 0:
                i-=1

            if i == 0:
                metadata['time_to_peak'] = np.nan
            else:
                metadata['time_to_peak'] = self.data.data[self.data.time_column][peak_idx] - self.data.data[self.data.time_column][i]

            i = peak_idx
            while self.data.smooth[i] != diastolic_reference_at_peaks[i] and i < len(self.data.smooth) - 1:
                i+=1
            
            if i == len(self.data.smooth) - 1:
                metadata['time_to_relaxation'] = np.nan
            else:
                metadata['time_to_relaxation'] = self.data.data[self.data.time_column][i] - self.data.data[self.data.time_column][peak_idx]

            self.peak_objects.append(metadata)



            