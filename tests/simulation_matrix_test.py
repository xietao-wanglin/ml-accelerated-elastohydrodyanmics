from unittest import TestCase

import numpy as np

from simulation_matrix import Simulation, FlowLibrary

class TestFlowMatrix(TestCase):

    def test_3d_animation_single_dumbbell(self):
        r = np.array([[0, 0.01, 0.9], 
                      [0, -0.01, 0.9], 
                      [0, 0.01, 0.5], 
                      [0, -0.01, 0.5], 
                      [0, 0.01, 1.3], 
                      [0, -0.01, 1.3], 
                      [0, 0.01, 1.7], 
                      [0, -0.01, 1.7],
                      [0, 0.01, 2.5], 
                      [0, -0.01, 2.5],])
        f = FlowLibrary()
        sim = Simulation(bead_pos=r, gravity=0.1, repulsion=1.0, repulsion_range=0.001, boundary=True, rest_length=0.01)
        sim.solve_dynamics(max_time=80.0, verbose=True, method='Radau')
        sim.create_3d_animation(domain=[[-0.5, -0.5, 0], [0.5, 0.5, 2.5]], arrow_size=0, same_size=False, filename='sedimentation_short.mp4')

    def test_2d_projection(self):
        r = np.array([[0, 0, 0.5], 
                      [0, 0, -0.5],])
        f = FlowLibrary()
        sim = Simulation(bead_pos=r, gravity=0, repulsion=0, repulsion_range=0.15, boundary=False, bg_flow=f.shear_flow_3d)
        sim.solve_dynamics(max_time=10.0, verbose=True, method='Radau')
        sim.create_2d_projection(domain=[[-1, -1, -1], [1, 1, 1]], grid_points=20)
