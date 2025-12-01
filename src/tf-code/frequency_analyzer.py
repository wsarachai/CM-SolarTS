import numpy as np
import tensorflow as tf

class FrequencyAnalyzer:
    def __init__(self, signal, sample_period):
        self.signal = signal
        self.sample_period = sample_period
        self.n_samples = len(signal)
        self.fft = None
        self.amplitude_spectrum = None
        self.frequencies = None
        self.peaks = None
        self.peak_frequencies = None
        self.peak_periods = None

    @staticmethod
    def manual_rfftfreq(n_samples, d):
        sample_rate = 1.0 / d
        num_freq_bins = n_samples // 2 + 1
        freq_increment = sample_rate / n_samples
        frequencies = np.arange(num_freq_bins) * freq_increment
        return frequencies

    @staticmethod
    def find_peaks_manual(x, min_prominence_ratio=0.01):
        peaks = []
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1]:
                peaks.append(i)
        max_val = x.max()
        threshold = max_val * min_prominence_ratio
        peaks = [p for p in peaks if x[p] > threshold]
        return np.array(peaks)

    def analyze(self):
        self.fft = tf.signal.rfft(self.signal)
        self.amplitude_spectrum = np.abs(self.fft.numpy())
        self.frequencies = self.manual_rfftfreq(self.n_samples, self.sample_period)
        peaks = self.find_peaks_manual(self.amplitude_spectrum)
        f_per_dataset = np.arange(0, len(self.fft))
        years_per_dataset = self.n_samples / (self.sample_period)
        f_per_year = f_per_dataset / years_per_dataset
        peak_frequencies = f_per_year[peaks]
        peak_periods = 1 / peak_frequencies
        sorted_indices = np.argsort(-self.amplitude_spectrum[peaks])
        self.peaks = peaks[sorted_indices]
        self.peak_frequencies = peak_frequencies[sorted_indices]
        self.peak_periods = peak_periods[sorted_indices]
        return self.peak_frequencies, self.peak_periods