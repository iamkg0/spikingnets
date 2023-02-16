from layers import *

class Synapse:
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.weights = np.random.uniform(low=.4, high=.6, size=(len(presynaptic), len(postsynaptic)))
        
    def __getitem__(self, presynaptic_idx, postsynaptic_idx):
        return self.weights[presynaptic_idx, postsynaptic_idx]

    def normalize(self):
        self.weights = (self.weights - np.min(self.weights)) / (np.max(self.weights) - np.min(self.weights))

    def forward(self):
        self.presynaptic.forward()
        currents = np.dot(self.presynaptic.propagate(), self.weights)
        self.postsynaptic.apply_current(currents)
        #self.postsynaptic.forward()

    def STDP(self, learning_rate=.1, assymetry = 5):
        if self.presynaptic.spiked.any() or self.postsynaptic.spiked.any():
            presynaptic_impulses = self.postsynaptic.impulses * self.presynaptic.spiked
            presynaptic_impulses = np.tile(presynaptic_impulses, len(self.postsynaptic)).reshape(self.weights.shape)
            postsynaptic_impulses = self.presynaptic.impulses * self.postsynaptic.spiked
            postsynaptic_impulses = np.tile(postsynaptic_impulses, len(self.presynaptic)).reshape(len(self.postsynaptic), len(self.presynaptic)).T
            self.weights -= presynaptic_impulses * self.weights * learning_rate
            self.weights += postsynaptic_impulses * (1 - self.weights) * learning_rate * assymetry