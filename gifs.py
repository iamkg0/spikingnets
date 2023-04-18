from layers import *
from synaptics import *
from visual_env import *
from stats import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PARAMETERS OF NETWORK:
retina_size = (1, 10)
resolution = .1
input_layer_size = retina_size[0] * retina_size[1]
output_layer_size = 1
tau = 30
triplet_tau = 0
preset = 'RS'
g_strength = 200

layer0 = IzhikevichLayer(size=input_layer_size, resolution=resolution, tau=tau, preset=preset, noize=1)
layer0.transmitter_impact = g_strength
layer1 = IzhikevichLayer(size=output_layer_size, resolution=resolution, tau=tau, preset=preset, noize=1)
synapse = Synapse(layer0, layer1)
rates0 = rate_capture(layer0)
rates1 = rate_capture(layer1)
synapse.load_weights(name='checkpoints\save10_1by3.npy')

#weights = np.zeros(input_layer_size*output_layer_size).reshape(1, input_layer_size, output_layer_size)
pre_v = np.zeros(input_layer_size).reshape(1, input_layer_size)
pre_syn = np.zeros(input_layer_size).reshape(1, input_layer_size)
post_v = np.zeros(output_layer_size).reshape(1, output_layer_size)
post_syn = np.zeros(output_layer_size).reshape(1, output_layer_size)
rates0.reset()
rates1.reset()

vis = retina(size=retina_size)
pattern = np.ones(3).reshape(1,3) * 25
vis.add_object(pattern)
vis.set_position_lazy(x='right', y='centered')
state = vis.show_current_state()
vis_hist = []

g_strength = 85 #85
layer0.transmitter_impact = g_strength

t = 500
time = int(t / resolution)
lr = .01
alpha = 5
pattern_delay = int(4/ resolution)
direction = 'left'
gather_data = True
learn = False


for i in range(time):
    synapse.forward()
    picture = vis.tick(delay=pattern_delay, move_direction=direction, noize_density=.1, noize_acceleration=7, rest=int(220/resolution))
    layer0.apply_current(picture.flatten())
    layer1.forward()
    rates0.accumulate_spikes()
    rates1.accumulate_spikes()
    if learn:
        synapse.STDP(learning_rate=lr, assymetry=alpha)
    #synapse.STDP(learning_rate=lr, assymetry=alpha)
    if gather_data:
        #weights = np.append(weights, np.array([synapse.weights]), axis=0)
        pre_v = np.append(pre_v, np.array([layer0.v]), axis=0)
        pre_syn = np.append(pre_syn, np.array([layer0.impulses]), axis=0)
        post_v = np.append(post_v, np.array([layer1.v]), axis=0)
        post_syn = np.append(post_syn, np.array([layer1.impulses]), axis=0)
        vis_hist.append(picture.flatten())
        #print(picture)
layer0.instant_rest()
layer1.instant_rest()




'''
for i in range(pre_v.shape[1]):
    temp = np.array(pre_v[:, i]).flatten()
    plt.plot(x_scale[1:], temp[1:])
plt.show()
for i in range(post_v.shape[1]):
    temp = np.array(post_v[:, i]).flatten()
    plt.plot(x_scale[1:], temp[1:])
plt.show()
for i in range(pre_syn.shape[1]):
    temp = np.array(pre_syn[:, i]).flatten()
    plt.plot(x_scale[1:], temp[1:])
plt.show()
for i in range(post_syn.shape[1]):
    temp = np.array(post_syn[:, i]).flatten()
    plt.plot(x_scale[1:], temp[1:])
plt.show()
'''




plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
x_scale = np.arange(int(pre_v.shape[0])-1) * resolution
x = post_v[1:,0]
line, = ax.plot(x_scale, x)


def animate(i):
    i *= 25
    line.set_data(x_scale[0:i+1], x[0:i+1])
    #line.set_ydata(x[:i+1])  # update the data.
    return line,


anim = animation.FuncAnimation(fig, animate, interval=1, blit=True, frames=200)
plt.title('Output neuron')
anim.save('Neu_U.gif', writer='imagemagick', fps=30)
#plt.show()

#or i in range(len(vis_hist)):
    #print(vis_hist[i])
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots()
#print(picture.shape, vis_hist[0].shape)
def update(i):
    i *= 25
    im_normed = [vis_hist[i]]
    ax.imshow(im_normed)
    ax.set_axis_off()

anim = animation.FuncAnimation(fig, update, frames=200, interval=1)
plt.title('Unfamiliar pattern')
#anim.save('patternl.gif', writer='pillow', fps=100)
#plt.show()
#writervideo = animation.FFMpegWriter(fps=60)
anim.save('patternU.gif', writer='imagemagick', fps=30)
plt.close()
