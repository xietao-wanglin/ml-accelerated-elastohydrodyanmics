import time

import numpy as np

from simulation import Simulation

if __name__ == '__main__':

    num_of_beads = np.array([2**x for x in range(1, 16)])
    n_replic = 5
    times = np.zeros((num_of_beads.shape[0], n_replic))
    for i, n in enumerate(num_of_beads):
        for replic in range(n_replic):
            r = np.random.uniform(low=-1, high=1, size=(n, 2))
            start = time.time()
            sim = Simulation(bead_pos=r)
            sim.solve_dynamics(max_time=10.0, verbose=True)
            end = time.time()
            times[i, replic] = end-start
        print(f'Took {times[i].mean():2f} to simulate {n} beads.\n')
