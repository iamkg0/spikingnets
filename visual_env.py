import numpy as np

class retina:
    def __init__(self, size=(5,5)):
        self.lid = np.zeros(size)
        self.buffer_lid = None
        self.object = None
        self.obj_position = None
        self.delay_counter = 0
        self.directions = {'right': [0, 1],
                           'left': [0, -1],
                           'up': [-1, 0],
                           'down': [1, 0]}
        self.lid_area = None

    def add_object(self, object, position=[2,0]):
        self.object = object
        self.obj_position = position
        self.buffer_lid = np.zeros((self.lid.shape[0]+self.object.shape[0]*2, self.lid.shape[1]+self.object.shape[1]*2))
        self.lid_area = self.object.shape, (self.object.shape[0]+self.lid.shape[0], self.object.shape[1]+self.lid.shape[1]) # a, c, b, d
        a, b, c, d = self.determine_position()
        self.buffer_lid[a:b, c:d] = self.object

    def determine_position(self):
        ax0_end = self.obj_position[0] + self.object.shape[0]
        ax1_end = self.obj_position[1] + self.object.shape[1]
        if self.obj_position[0] < 0: # too much to the left
            self.obj_position[0] = self.object.shape[0]+self.lid.shape[0]
            ax0_end = self.obj_position[0] + self.object.shape[0]
        if self.obj_position[1] < 0: # too much to the top
            self.obj_position[1] = self.object.shape[1]+self.lid.shape[1]
            ax1_end = self.obj_position[1] + self.object.shape[1]
        if ax0_end > self.buffer_lid.shape[0]: # too much to the right
            self.obj_position[0] = 0
            ax0_end = self.obj_position[0] + self.object.shape[0]
        if ax1_end > self.buffer_lid.shape[1]: # too much to the bottom
            self.obj_position[1] = 0
            ax1_end = self.obj_position[1] + self.object.shape[1]
        return self.obj_position[0], ax0_end, self.obj_position[1], ax1_end

    def move_object(self, direction='right'):
        self.obj_position += np.array(self.directions[direction])
        self.buffer_lid *= 0
        a, b, c, d = self.determine_position()
        self.buffer_lid[a:b, c:d] = self.object
    
    def tick(self, delay=0, move_direction='right'):
        if self.delay_counter >= delay:
            self.delay_counter = 0
            self.move_object(direction=move_direction)
        else:
            self.delay_counter += 1
        return self.buffer_lid[self.lid_area[0][0]:self.lid_area[1][0], self.lid_area[0][1]:self.lid_area[1][1]]