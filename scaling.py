import time

import numpy as np

from simulation import Simulation

if __name__ == '__main__':

    num_of_beads = np.array([2**x for x in range(1, 14)])
    n_replic = 5
    times = np.zeros((num_of_beads.shape[0], n_replic))
    filename = 'scaling.dat'
    with open(filename, 'w') as f:
        f.close()
    for i, n in enumerate(num_of_beads):
        for replic in range(n_replic):
            r = np.random.uniform(low=-1, high=1, size=(n, 2))
            start = time.time()
            sim = Simulation(bead_pos=r)
            sim.solve_dynamics(max_time=10.0, method='Radau', verbose=False)
            end = time.time()
            times[i, replic] = end-start
        with open(filename, 'ab') as f:
            np.savetxt(f, np.atleast_2d(times[i]), delimiter=',', newline = '\n')
        print(f'Took {times[i].mean():2f} to simulate {n} beads.\n')
