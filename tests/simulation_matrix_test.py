import unittest
from unittest import TestCase

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d

from simulation_matrix import Simulation, FlowLibrary

class TestFlowMatrix(TestCase):

    def test_3d_animation_single_dumbbell(self):
        r = np.array([[0, 0.5, 3], 
                      [0, -0.5, 3],
                      [0, 0.5, 5], 
                      [0, -0.5, 5],
                      [0, 0.5, 2], 
                      [0, -0.5, 2],
                      [0, 0.5, 4], 
                      [0, -0.5, 4],])
        sim = Simulation(bead_pos=r, gravity=0.1, repulsion=0.2, repulsion_range=0.15, boundary=True, rest_length=0.05)
        sim.solve_dynamics(max_time=100.0, verbose=True, method='Radau')
        sim.create_3d_animation(domain=[[-1, -1, 0], [1, 1, 10]], arrow_size=0.5, same_size=True)
