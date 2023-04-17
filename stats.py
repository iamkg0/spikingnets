import numpy as np

class rate_capture:
    def __init__(self, layer):
        self.layer = layer
        self.spikes = np.zeros(len(self.layer))

    def accumulate_spikes(self, time=1000):
        self.spikes += self.layer.spiked

    def compute_spike_rates(self, time, interval=1000):
        denominator = time / interval
        return self.spikes / denominator
    
    def reset(self):
        self.spikes *= 0