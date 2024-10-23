import time
from typing import Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

class Simulation:

    def __init__(self, bead_pos: Optional[np.ndarray] = None,
                 mu: Optional[float] = 1.0,
                 k: Optional[float] = 0.1,
                 eps: Optional[float] = 0.15,
                 rest_length: Optional[float] = 0.25,
                 shear: Optional[float] = 1.0) -> None:
        """
        Initialises the Simulation class.

        Parameters:
        bead_pos (ndarray, optional): 
            Position of beads at the start, default is array([[0, 0.5], [0, -0.5]]).
            The rows are given by coordinates of each of the beads. Must have an even number of rows
        mu (float, optional): dynamic viscosity, default is 1.0.
        k (float, optional): spring constant of the dumbbells, default is 0.1
        eps (float, optional): value of epsilon used in the the regular stokeslet, default is 0.15.
        rest_length (float, optional): rest length of the dumbbell, default is 0.25.
        shear (float, optional): background shear strength, default is 1.0.
        """
        
        self.mu = mu
        self.k = k
        self.eps = eps
        self.rest_length = rest_length
        self.shear = shear

        if bead_pos is None:
            self.bead_pos = np.array([[0, 0.5], [0, -0.5]])
        else:
            self.bead_pos = np.atleast_2d(bead_pos)
        
        if (self.bead_pos.shape[0] % 2) == 1:
            raise ValueError('Number of beads must be even.')
        
        self.dim = self.bead_pos.shape[1]
        self.bead_sol = None

    def fluid_flow(self, x, y):
        """
        Returns the current fluid flow at position (x, y).
        """
    
        xr = self.bead_pos[1::2] - self.bead_pos[::2]
        dists = np.sqrt(xr[:, 0] * xr[:, 0] + xr[:, 1] * xr[:, 1])
            
        x = np.asarray(x)
        y = np.asarray(y)

        rn = np.sqrt((x[..., np.newaxis] - self.bead_pos[:, 0])**2 + 
                    (y[..., np.newaxis] - self.bead_pos[:, 1])**2)

        const = self.k*(dists-self.rest_length)/(8*np.pi*self.mu*self.rest_length*dists)

        s1x = (xr[:, 0] * (rn[..., ::2]**2 + 2 * self.eps**2) + 
            (x[..., np.newaxis] - self.bead_pos[::2, 0]) * 
            ((x[..., np.newaxis] - self.bead_pos[::2, 0]) * xr[:, 0] + 
                (y[..., np.newaxis] - self.bead_pos[::2, 1]) * xr[:, 1])) / (rn[..., ::2]**2 + self.eps**2)**(3/2)

        s1y = (xr[:, 1] * (rn[..., ::2]**2 + 2 * self.eps**2) + 
            (y[..., np.newaxis] - self.bead_pos[::2, 1]) * 
            ((x[..., np.newaxis] - self.bead_pos[::2, 0]) * xr[:, 0] + 
                (y[..., np.newaxis] - self.bead_pos[::2, 1]) * xr[:, 1])) / (rn[..., ::2]**2 + self.eps**2)**(3/2)

        s2x = (xr[:, 0] * (rn[..., 1::2]**2 + 2 * self.eps**2) + 
            (x[..., np.newaxis] - self.bead_pos[1::2, 0]) * 
            ((x[..., np.newaxis] - self.bead_pos[1::2, 0]) * xr[:, 0] + 
                (y[..., np.newaxis] - self.bead_pos[1::2, 1]) * xr[:, 1])) / (rn[..., 1::2]**2 + self.eps**2)**(3/2)

        s2y = (xr[:, 1] * (rn[..., 1::2]**2 + 2 * self.eps**2) + 
            (y[..., np.newaxis] - self.bead_pos[1::2, 1]) * 
            ((x[..., np.newaxis] - self.bead_pos[1::2, 0]) * xr[:, 0] + 
                (y[..., np.newaxis] - self.bead_pos[1::2, 1]) * xr[:, 1])) / (rn[..., 1::2]**2 + self.eps**2)**(3/2)

        u = (s1x * const).sum(axis=-1) - (s2x * const).sum(axis=-1) + self.shear * y
        v = (s1y * const).sum(axis=-1) - (s2y * const).sum(axis=-1)
        return u, v
    
    def bead_dynamics(self, t, positions):
        """
        Creates ODE system wrapper for SciPy's solve_ivp routine.
        """

        num_of_beads = int(positions.shape[0]/2)
        positions = positions.reshape(num_of_beads, 2)
        u_array = np.zeros(2*num_of_beads)
            
        for n in range(num_of_beads):
        
            xr = positions[1::2] - positions[::2]
            dists = np.sqrt(xr[:, 0] * xr[:, 0] + xr[:, 1] * xr[:, 1])
            x = positions[n, 0]
            y = positions[n, 1]

            x = np.asarray(x)
            y = np.asarray(y)

            rn = np.sqrt((x[..., np.newaxis] - positions[:, 0])**2 + 
                        (y[..., np.newaxis] - positions[:, 1])**2)

            const = self.k * (dists - self.rest_length) / (8 * np.pi * self.mu * self.rest_length * dists)

            s1x = (xr[:, 0] * (rn[..., ::2]**2 + 2 * self.eps**2) + 
                (x[..., np.newaxis] - positions[::2, 0]) * 
                ((x[..., np.newaxis] - positions[::2, 0]) * xr[:, 0] + 
                    (y[..., np.newaxis] - positions[::2, 1]) * xr[:, 1])) / (rn[..., ::2]**2 + self.eps**2)**(3/2)

            s1y = (xr[:, 1] * (rn[..., ::2]**2 + 2 * self.eps**2) + 
                (y[..., np.newaxis] - positions[::2, 1]) * 
                ((x[..., np.newaxis] - positions[::2, 0]) * xr[:, 0] + 
                    (y[..., np.newaxis] - positions[::2, 1]) * xr[:, 1])) / (rn[..., ::2]**2 + self.eps**2)**(3/2)

            s2x = (xr[:, 0] * (rn[..., 1::2]**2 + 2 * self.eps**2) + 
                (x[..., np.newaxis] - positions[1::2, 0]) * 
                ((x[..., np.newaxis] - positions[1::2, 0]) * xr[:, 0] + 
                    (y[..., np.newaxis] - positions[1::2, 1]) * xr[:, 1])) / (rn[..., 1::2]**2 + self.eps**2)**(3/2)

            s2y = (xr[:, 1] * (rn[..., 1::2]**2 + 2 * self.eps**2) + 
                (y[..., np.newaxis] - positions[1::2, 1]) * 
                ((x[..., np.newaxis] - positions[1::2, 0]) * xr[:, 0] + 
                    (y[..., np.newaxis] - positions[1::2, 1]) * xr[:, 1])) / (rn[..., 1::2]**2 + self.eps**2)**(3/2)

            u = (s1x * const).sum(axis=-1) - (s2x * const).sum(axis=-1) + self.shear * y
            v = (s1y * const).sum(axis=-1) - (s2y * const).sum(axis=-1)
            
            u_array[2*n] = u
            u_array[2*n+1] = v
        return u_array
    
    def solve_dynamics(self, max_time: Optional[float] = 1.0, 
                       verbose: Optional[bool] = False) -> None:
        """
        Solves ODE system using solve_ivp
        """

        if verbose:
            start = time.time()
            print('Solving bead dynamics... \n')
        self.bead_sol = solve_ivp(self.bead_dynamics, 
                                  [0, max_time], 
                                  y0 = np.ravel(self.bead_pos), 
                                  dense_output=True, 
                                  method='RK45')

        if verbose:
            end = time.time()
            print(f'Solved bead dyanmics, took {end-start:2f} seconds.')

    def create_animation(self, domain: Optional[list] = None, 
                         grid_points: Optional[int] = 100, 
                         n_timesteps: Optional[int] = 100, 
                         filename: Optional[str] = None) -> None:
        """
        Creates animation of beads and fluid flow.

        Parameters:
        domain (list, optional): the space domain to visualise, default is [[-1, -1], [1, 1]].
        grid_points (int, optional): number of evenly spaced grid points used to calculate the flow, default is 100.
        n_timesteps (int, optional): number of evenly spaced timesteps to calculate the evolution, default is 100.
        filename (str, optional): None to not create a video file, 
        otherwise string will be used as filename (extension and video format not included). 
        """

        if self.bead_sol is None:
            raise ValueError('Solve bead dynamics first')
        
        if domain is None:
            domain = [[-1, -1], [1, 1]]

        x_flat = np.linspace(domain[0][0], domain[1][0], grid_points)
        y_flat = np.linspace(domain[0][1], domain[1][1], grid_points)
        x, y = np.meshgrid(x_flat, y_flat)
        u, v = self.fluid_flow(x, y)
        num_of_dumbbells = int(self.bead_pos.shape[0]/2)

        f = plt.figure(figsize=(6, 5))
        ax = f.add_subplot(111)
        ax.set_xlim(domain[0][0], domain[1][0])
        ax.set_ylim(domain[0][1], domain[1][1],)
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(r'Time: 0.0')
        self.stream = ax.streamplot(x, y, u, v, color='red')
        self.heatmap = ax.pcolormesh(x, y, np.sqrt(u*u+v*v), alpha=1, cmap='PiYG',
                                      norm=colors.LogNorm(vmin=0.001, vmax=100))
        lines = [None]*num_of_dumbbells
        for d in range(num_of_dumbbells):
            lines[d], = ax.plot(self.bead_pos[2*d:2*d+2][:, 0], self.bead_pos[2*d:2*d+2][:, 1],
                                 marker='o', lw=4)
        f.colorbar(self.heatmap, label=r'$|u| = \sqrt{\partial_t x_i \partial_t x_i}$')
        max_t = self.bead_sol.t[-1]
        t = np.linspace(0, max_t, n_timesteps)
        sol = self.bead_sol.sol(t)
        def update(fn):
            ax.set_title(fr'Time: {t[fn]:2f}')
            self.bead_pos = sol[:, fn].reshape(num_of_dumbbells*2, 2)
            u, v = self.fluid_flow(x, y)
            z = np.sqrt(u*u + v*v)
            self.heatmap.set_array(z)
            
            for d in range(num_of_dumbbells):
                lines[d].set_data(self.bead_pos[2*d:2*d+2][:, 0], self.bead_pos[2*d:2*d+2][:, 1])
            self.stream.lines.remove()
            for art in ax.get_children():
                if not isinstance(art, mpl.patches.FancyArrowPatch):
                    continue
                art.remove()
            
            self.stream = ax.streamplot(x, y, u, v, color='red')

        animation = FuncAnimation(f, update, interval=1, frames=n_timesteps)
        plt.show()
        if filename is not None:
            animation.save(f'./videos/{filename}')
