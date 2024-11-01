import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from simulation import Simulation, FlowLibrary

def flow_around_dumbbell():

    f, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

    sim = Simulation(bg_flow=FlowLibrary.shear_flow_2d)
    sim.solve_dynamics(max_time=10.0, verbose=False)
    x_flat = np.linspace(-1, 1, 100)
    y_flat = np.linspace(-1, 1, 100)
    X = np.meshgrid(x_flat, y_flat)
    U = sim.fluid_flow(X, 0)
    u, v = U[..., 0], U[..., 1]
    x, y = X

    max_t = sim.bead_sol.t[-1]
    t = np.linspace(0, max_t, 101)
    sol = sim.bead_sol.sol(t)

    ax[0].set_title(fr'Time: {t[0]:1f}')
    ax[0].plot(sim.bead_pos[:, 0], sim.bead_pos[:, 1],
                            marker='o', lw=4, color='green')
    ax[0].streamplot(x, y, u, v, color='red')
    ax[0].pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                    norm=colors.LogNorm(vmin=0.001, vmax=100))
    ax[0].set_xlim(-1, 1)
    ax[0].set_ylim(-1, 1)

    ax[1].set_title(fr'Time: {t[25]:1f}')
    sim.bead_pos = sol[:, 33].reshape(1*2, 2)
    U = sim.fluid_flow(X, t[25])
    u, v = U[..., 0], U[..., 1]

    ax[1].plot(sim.bead_pos[:, 0], sim.bead_pos[:, 1],
                            marker='o', lw=4, color='green')
    ax[1].streamplot(x, y, u, v, color='red')
    ax[1].pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                    norm=colors.LogNorm(vmin=0.001, vmax=100))
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(-1, 1)

    ax[2].set_title(fr'Time: {t[80]:1f}')
    sim.bead_pos = sol[:, 80].reshape(1*2, 2)
    U = sim.fluid_flow(X, t[80])
    u, v = U[..., 0], U[..., 1]

    ax[2].plot(sim.bead_pos[:, 0], sim.bead_pos[:, 1],
                            marker='o', lw=4, color='green')
    ax[2].streamplot(x, y, u, v, color='red')
    heatmap = ax[2].pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                    norm=colors.LogNorm(vmin=0.001, vmax=100))
    ax[2].set_xlim(-1, 1)
    ax[2].set_ylim(-1, 1)

    f.colorbar(mappable=heatmap, label=r'$|u| = \sqrt{u_i u_i}$')

    f.supxlabel(r'$x_1$')
    f.supylabel(r'$x_2$')
    f.tight_layout()

    plt.savefig('./figures/flow_around_dumbbell.pdf', bbox_inches='tight')
    plt.savefig('./figures/flow_around_dumbbell.png', bbox_inches='tight')
    plt.show()

def flow_multiple_dumbbells():

    f, ax = plt.subplots(1, 3, figsize=(12, 5), sharex=True, sharey=True)

    r = np.array([[ 0.72771121, +0.76493629],
            [ 0.03475821, -0.73586379],
            [ 0.83371936, +0.20788059],
            [ 0.13084262, -0.63344033],
            [-0.71030448, -0.8388744],
            [-0.28877452,  0.88086389]])
    
    sim = Simulation(bead_pos=r, bg_flow=FlowLibrary.shear_flow_2d)
    sim.solve_dynamics(max_time=10.0, verbose=False)
    x_flat = np.linspace(-2, 2, 100)
    y_flat = np.linspace(-2, 2, 100)
    X = np.meshgrid(x_flat, y_flat)
    U = sim.fluid_flow(X, 0)
    u, v = U[..., 0], U[..., 1]
    x, y = X

    max_t = sim.bead_sol.t[-1]
    t = np.linspace(0, max_t, 101)
    sol = sim.bead_sol.sol(t)

    ax[0].set_title(fr'Time: {t[0]:1f}')
    for i in range(sim.num_of_dumbbells):
        ax[0].plot(sim.bead_pos[2*i:2*i+2][:, 0], sim.bead_pos[2*i:2*i+2][:, 1],
                            marker='o', lw=4)
    ax[0].streamplot(x, y, u, v, color='red')
    ax[0].pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                    norm=colors.LogNorm(vmin=0.001, vmax=100))
    ax[0].set_xlim(-2, 2)
    ax[0].set_ylim(-2, 2)

    ax[1].set_title(fr'Time: {t[25]:1f}')
    sim.bead_pos = sol[:, 25].reshape(sim.num_of_dumbbells*2, 2)
    U = sim.fluid_flow(X, t[25])
    u, v = U[..., 0], U[..., 1]

    for i in range(sim.num_of_dumbbells):
        ax[1].plot(sim.bead_pos[2*i:2*i+2][:, 0], sim.bead_pos[2*i:2*i+2][:, 1],
                            marker='o', lw=4)
    ax[1].streamplot(x, y, u, v, color='red')
    ax[1].pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                    norm=colors.LogNorm(vmin=0.001, vmax=100))
    ax[1].set_xlim(-2, 2)
    ax[1].set_ylim(-2, 2)

    ax[2].set_title(fr'Time: {t[80]:1f}')
    sim.bead_pos = sol[:, 80].reshape(sim.num_of_dumbbells*2, 2)
    U = sim.fluid_flow(X, t[80])
    u, v = U[..., 0], U[..., 1]

    for i in range(sim.num_of_dumbbells):
        ax[2].plot(sim.bead_pos[2*i:2*i+2][:, 0], sim.bead_pos[2*i:2*i+2][:, 1],
                            marker='o', lw=4)
    ax[2].streamplot(x, y, u, v, color='red')
    heatmap = ax[2].pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                    norm=colors.LogNorm(vmin=0.001, vmax=100))
    ax[2].set_xlim(-2, 2)
    ax[2].set_ylim(-2, 2)

    f.colorbar(mappable=heatmap, label=r'$|u| = \sqrt{u_i u_i}$')

    f.supxlabel(r'$x_1$')
    f.supylabel(r'$x_2$')
    f.tight_layout()

    plt.savefig('./figures/flow_multiple_dumbbells.pdf', bbox_inches='tight')
    plt.savefig('./figures/flow_multiple_dumbbells.png', bbox_inches='tight')
    plt.show()

def simulation_scaling_implicit():

    data = np.loadtxt('./data/scaling.dat', delimiter=',')
    mean = data.mean(axis=1)
    error = data.std(axis=1)/data.shape[1]
    num_of_beads = np.array([2**x for x in range(1, 11)])/2
    f, ax = plt.subplots(1, 1)
    ax.errorbar(num_of_beads, mean, yerr=error, marker='o', lw=2, color='red', capsize=4, label='Computed runtime')
    ax.plot(num_of_beads[3:], (0.00005*num_of_beads**3)[3:], lw=2, ls='dashed', color='black', label=r'$O(N^3)$')
    ax.set_ylabel('Time, seconds')
    ax.set_xlabel('Number of dumbbells, $N$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(frameon=False)

    plt.savefig('./figures/simulation_scaling_implicit.pdf', bbox_inches='tight')
    plt.savefig('./figures/simulation_scaling_implicit.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':

    flow_around_dumbbell()
    flow_multiple_dumbbells()
    simulation_scaling_implicit()
