from simulation import Simulation

if __name__ == '__main__':

    sim = Simulation()
    sim.solve_dynamics(max_time=10.0, verbose=True)
    sim.create_animation()
