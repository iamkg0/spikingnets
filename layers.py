import numpy as np

class SNNLayer:
    def __init__(self, **kwargs):
        self.size = kwargs.get('size', 10)
        self.resolution = kwargs.get('resolution', .5)
        self.inhibitory = kwargs.get('inhibitory', False)
        self.ap_threshold = kwargs.get('ap_threshold', None)
        self.I = kwargs.get('I', 0)
        self.spiked = np.zeros(self.size)
        self.transmitter_impact = 1
        self.tau = kwargs.get('tau', 20)
        self.impulses = np.zeros(self.size)
        self.synaptic_output = kwargs.get('synaptic_output', True)
        self.spiked = np.zeros(self.size)
        self.impulses = np.zeros(self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.v[index]

    def apply_current(self, current):
        self.I = current

    def propagate(self):
        return self.spiked * self.transmitter_impact

    def spike_trace(self):
        self.impulses -= (self.impulses / self.tau) * self.resolution
        self.impulses += self.spiked
        return self.impulses

    def dynamics(self):
        return

    def forward(self):
        burst = self.dynamics()
        if self.synaptic_output:
            self.spike_trace()
        return burst





class IzhikevichLayer(SNNLayer):
    def __init__(self, size, **kwargs):
        super().__init__()
        '''
        Initializes layer of Izhikevich neurons:
        https://www.izhikevich.org/publications/spikes.htm
        Arguments:
            size - the number of neurons in layer, int
        **kwargs:
            resolution - size of dynamics step. The defalut value is 0.5
            inhibitory - if True, AP gives negative value, bool
            ap_threshold - the default threshold, on which AP is generated
            a, b, c, d - the parameters from original paper. Defaults correspond to RS configuration
            preset - the name of default preset from original paper, default is None
            noize - applies additional random uniform noize to potential dynamics
            tau - 
            synaptic_output - 
        '''
        self.size = size
        self.noize = kwargs.get('noize', 0)
        self.resolution = kwargs.get('resolution', .5)
        self.inhibitory = kwargs.get('inhibitory', False)
        if self.inhibitory == False:
            self.transmitter_impact = 1
        else:
            self.transmitter_impact = -1
        self.ap_threshold = kwargs.get('ap_threshold', 30)
        self.tau = kwargs.get('tau', 30)
        self.synaptic_output = kwargs.get('synaptic_output', True)
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        preset = kwargs.get('preset', None)
        param_list = [[0.02, 0.2, -65, 8],
						[0.02, 0.2, -55, 4],
						[0.02, 0.2, -50, 2],
						[0.1, 0.2, -65, 2],
						[0.02, 0.25, -65, 0.05],
						[0.1, 0.3, -65, 2],
						[0.02, 0.25, -65, 2],
						[kwargs.get('a', .02), kwargs.get('b', .2), kwargs.get('c', -65), kwargs.get('d', 2)]]
        idx = preset_list.index(preset)
        assert preset in preset_list, f'Preset {preset} does not exist! Use one from {preset_list}'
        self.a = param_list[idx][0]
        self.b = param_list[idx][1]
        self.c = param_list[idx][2]
        self.d = param_list[idx][3]
        self.v = np.ones(self.size) * self.c
        self.u = np.ones(self.size) * self.b * self.v
        self.spiked = np.zeros(self.size)
        self.impulses = np.zeros(self.size)

    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I) + np.random.uniform(-self.noize, self.noize)
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        # AP AND RECOVERY:
        self.spiked = self.v >= self.ap_threshold
        didnt_spike = self.v < self.ap_threshold
        temp_recovered = self.spiked * self.c, self.spiked * (self.u + self.d)
        temp_working = didnt_spike * self.v, didnt_spike * self.u
        self.v = temp_recovered[0] + temp_working[0]
        self.u = temp_recovered[1] + temp_working[1]
        return self.spiked.astype(float)





class IandFLayer(SNNLayer):
    def __init__(self, size, **kwargs):
        self.size = size
        self.noize = kwargs.get('noize', 0)
        self.resolution = kwargs.get('resolution', 1)
        self.inhibitory = kwargs.get('inhibitory', False)
        if self.inhibitory == False:
            self.transmitter_impact = 1
        else:
            self.transmitter_impact = -1
        self.ap_threshold = kwargs.get('ap_threshold', 30)
        self.tau = kwargs.get('tau', 30)
        self.synaptic_output = kwargs.get('synaptic_output', True)
        self.spiked = np.zeros(self.size)
        self.impulses = np.zeros(self.size)
        self.v = np.zeros(size)
        self.I = np.zeros(size)

    def dynamics(self):
        self.v += (-self.v / self.tau + self.I) *self.resolution
        # AP AND RECOVERY:
        self.spiked = self.v >= self.ap_threshold
        didnt_spike = self.v < self.ap_threshold
        temp_recovered = self.spiked * 0
        temp_working = didnt_spike * self.v
        self.v = temp_recovered + temp_working
        return self.spiked.astype(float)