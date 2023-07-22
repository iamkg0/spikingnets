from layers import *

class Synapse:
    def __init__(self, presynaptic, postsynaptic, **kwargs):
        self.presynaptic = presynaptic
        self.postsynaptic = postsynaptic
        self.weights = np.random.uniform(low=.4, high=.6, size=(len(presynaptic), len(postsynaptic)))
        self.modification = kwargs.get('modification', 1)
        
    def __getitem__(self, presynaptic_idx, postsynaptic_idx):
        return self.weights[presynaptic_idx, postsynaptic_idx]
    
    def weights_init(self, mode='all_05'):
        mode_case = {'all_05': self.weights_all_05}
        mode_case[mode]()

    def weights_all_05(self):
        self.weights *= 0
        self.weights += .5

    def normalize(self):
        self.weights = (self.weights - np.min(self.weights)) / (np.max(self.weights) - np.min(self.weights))

    def forward(self):
        self.presynaptic.forward()
        currents = np.dot(self.presynaptic.propagate() * self.modification, self.weights)
        self.postsynaptic.apply_current(currents)

    def STDP(self, learning_rate=.1, assymetry = 5):
        if self.presynaptic.spiked.any() or self.postsynaptic.spiked.any():
            presynaptic = np.tile(self.presynaptic.impulses, len(self.postsynaptic)).reshape(len(self.postsynaptic), len(self.presynaptic))
            postsynaptic = np.tile(self.postsynaptic.impulses, len(self.presynaptic)).reshape(self.weights.shape)
            postsynaptic_impulses = presynaptic.T * np.array([self.postsynaptic.spiked])           
            presynaptic_impulses = postsynaptic.T * np.array([self.presynaptic.spiked])
            self.weights += postsynaptic_impulses * (1 - self.weights) * learning_rate
            self.weights -= presynaptic_impulses.T * self.weights * learning_rate * assymetry
            
    def get_connection_info(self, pre_id=None, post_id=None):
        # This function will get sense someday
        return self.weights[pre_id, post_id]
    
    def save_weights(self, name='test_checkpoint.npy'):
        with open(name, 'wb') as f:
            np.save(f, self.weights)

    def load_weights(self, name='test_checkpoint.npy'):
        with open(name, 'rb') as f:
            self.weights = np.load(f)




class Conv:
    # Doesnt work yet. Didn't need it in diploma project
    # I will implement it later
    def __init__(self, kernel_size=3, core_size=1, inh_impact=-1, exc_impact=1):
        self.kernel_size = kernel_size
        self.core_size = core_size
        self.kernel = np.ones((kernel_size, kernel_size)) * inh_impact
        self.kernel[1,1] = exc_impact

    def forward(self, inp):
        temp = []
        for i in range(inp.shape[1] - self.kernel_size+1):
            for j in range(inp.shape[0] - self.kernel_size+1):
                receptive_field = np.sum(np.multiply(inp[i:self.kernel_size+i, j:self.kernel_size+j], self.kernel))
                temp.append(receptive_field)
        return np.array(temp)