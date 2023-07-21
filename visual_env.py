import numpy as np

class motion_pic:
    def __init__(self, size=(5,5)):
        self.size = size
        self.buffer_screen = None
        self.object = None
        self.obj_position = None
        self.delay_counter = 0
        self.directions = {'right': [0, 1],
                           'left': [0, -1],
                           'up': [-1, 0],
                           'down': [1, 0]}
        self.border_reached = False
        self.rest_timer = 0
        self.visual_area = None

    def add_object(self, object, position=[0,0]):
        self.object = object
        self.obj_position = position
        self.buffer_screen = np.zeros((self.size[0]+self.object.shape[0]*2, self.size[1]+self.object.shape[1]*2))
        self.visual_area = self.object.shape, (self.object.shape[0]+self.size[0], self.object.shape[1]+self.size[1]) # a, c, b, d
        a, b, c, d = self.determine_position()
        self.buffer_screen[a:b, c:d] = self.object

    def determine_position(self):
        ax0_end = self.obj_position[0] + self.object.shape[0]
        ax1_end = self.obj_position[1] + self.object.shape[1]
        if self.obj_position[0] < 0: # too much to the left
            self.border_reached = True
            self.obj_position[0] = self.object.shape[0]+self.size[0]
            ax0_end = self.obj_position[0] + self.object.shape[0]
        if self.obj_position[1] < 0: # too much to the top
            self.border_reached = True
            self.obj_position[1] = self.object.shape[1]+self.size[1]
            ax1_end = self.obj_position[1] + self.object.shape[1]
        if ax0_end > self.buffer_screen.shape[0]: # too much to the right
            self.border_reached = True
            self.obj_position[0] = 0
            ax0_end = self.obj_position[0] + self.object.shape[0]
        if ax1_end > self.buffer_screen.shape[1]: # too much to the bottom
            self.border_reached = True
            self.obj_position[1] = 0
            ax1_end = self.obj_position[1] + self.object.shape[1]
        return self.obj_position[0], ax0_end, self.obj_position[1], ax1_end
    
    def gain_noise(self, noise_density=.2, noise_acceleration=1):
        self.buffer_screen[self.visual_area[0][0]:self.visual_area[1][0],
                           self.visual_area[0][1]:self.visual_area[1][1]] = np.random.choice(a=[1*noise_acceleration,0], size=self.size, p=[noise_density, 1-noise_density])

    def move_object(self, direction='right', noise_density=.2, noise_acceleration=1):
        self.obj_position += np.array(self.directions[direction])
        self.buffer_screen *= 0
        a, b, c, d = self.determine_position()
        self.gain_noise(noise_density=noise_density, noise_acceleration=noise_acceleration)
        self.buffer_screen[a:b, c:d] = self.object
    
    def tick(self, delay=0, move_direction='right', noise_density=.2, noise_acceleration=1, rest=0):
        if self.border_reached:
            self.buffer_screen *= 0
            self.rest_timer += 1
            if self.rest_timer >= rest:
                self.gain_noise(noise_density=noise_density, noise_acceleration=noise_acceleration)
                self.border_reached = False
                self.rest_timer = 0
        else:
            if self.delay_counter >= delay:
                self.delay_counter = 0
                self.move_object(direction=move_direction, noise_density=noise_density, noise_acceleration=noise_acceleration)
                # For lazy debug
                #print(self.buffer_screen[self.visual_area[0][0]:self.visual_area[1][0], self.visual_area[0][1]:self.visual_area[1][1]])
            else:
                self.delay_counter += 1
        return self.buffer_screen[self.visual_area[0][0]:self.visual_area[1][0], self.visual_area[0][1]:self.visual_area[1][1]]
    
    def show_current_state(self):
        return self.buffer_screen[self.visual_area[0][0]:self.visual_area[1][0], self.visual_area[0][1]:self.visual_area[1][1]]
    
    def set_position_lazy(self, x='centered', y='centered', noise_density=0, noise_acceleration=0):
        y_positions = {'centered': int((self.object.shape[0] + self.size[0]) / 2),
                       'top': 0,
                       'bottom': self.size[0] + self.object.shape[0]}
        x_positions = {'centered': int((self.object.shape[1] + self.size[1]) / 2),
                       'left': 0,
                       'right': self.size[1] + self.object.shape[1]}

        self.buffer_screen *= 0
        a = x_positions[x]
        b = a + self.object.shape[1]
        c = y_positions[y]
        d = c + self.object.shape[0]
        self.obj_position = [c, a]
        self.gain_noise(noise_density=noise_density, noise_acceleration=noise_acceleration)
        self.buffer_screen[c:d, a:b] = self.object
        self.rest_timer = 0
        self.border_reached = False

    def static_with_noise(self, noise_density=.2, noise_acceleration=1, noise_length=0):
        # Doesnt work yet, just a placeholder
        if self.rest_timer > noise_length:
            self.rest_timer = 0
            self.gain_noise(noise_density=noise_density, noise_acceleration=noise_acceleration)
