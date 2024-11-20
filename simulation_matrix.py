import time
from typing import Optional, Callable

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
                 gravity: Optional[float] = 0.0,
                 repulsion: Optional[float] = 0.0,
                 repulsion_range: Optional[float] = 1.0,
                 bg_flow: Optional[Callable] = lambda _, __: 0,
                 boundary: Optional[bool] = False) -> None:
        """
        Initialises the Simulation class.

        Parameters
        ----------
        bead_pos: ndarray, optional 
            Position of beads at the start, default is None.
            The rows are given by coordinates of each of the beads. Must have an even number of rows and three columns.
        mu: float, optional
            Dynamic viscosity, default is 1.0.
        k: float, optional
            Spring constant of the dumbbells, default is 0.1
        eps: float, optional
            Value of epsilon used in the the regular stokeslet, default is 0.15.
        rest_length: float, optional 
            Rest length of the dumbbell, default is 0.25.
        gravity: float, optional
            Strength of the gravitational weight, default is 0.0.
        repulsion: float, optional
            Strength of repulsive force between beads, default is 0.0.
        repulsion_range: float, optional
            Parameter that controls the range of the repulsive force between beads, default is 1.0. Must be greater than 0.0
        bg_flow: Callable, optional
            An arbitrary background flow added at the end, default is lambda _, __: 0.
        boundary: bool, optional
            Set to true to introduce a plane boundary at z = 0, default is False.
        """
        
        self.mu = mu
        self.k = k
        self.eps = eps
        self.rest_length = rest_length
        self.gravity = gravity
        self.repulsion = repulsion
        self.bg_flow = bg_flow
        self.bead_sol = None
        self.boundary = boundary
        self.repulsion_range = repulsion_range
        
        if bead_pos is None:
            self.bead_pos = np.array([[0, 0.5, 0], [0, -0.5, 0]])
        else:
            self.bead_pos = bead_pos

        self.dim = 3
        if (self.bead_pos.shape[0] % 2) == 1:
            raise ValueError('Number of beads must be even.')
        
        self.num_of_dumbbells = int(self.bead_pos.shape[0]/2)
        self.num_of_beads = int(2*self.num_of_dumbbells)

    def regularised_stokeslet(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        x = np.atleast_2d(x)

        eps2 = self.eps*self.eps
        dx = x[:, 0] - y[0]
        dy = x[:, 1] - y[1]
        dz = x[:, 2] - y[2]
        
        # Reglarised stokeslet
        r2 = dx*dx+dy*dy+dz*dz
        reps = (r2 + eps2)**1.5
        reps_p = r2 + 2*eps2
        
        Gxx = (reps_p + dx*dx)/reps
        Gxy = dx*dy/reps
        Gxz = dx*dz/reps
        Gyy = (reps_p + dy*dy)/reps
        Gyz = dy*dz/reps
        Gzz = (reps_p + dz*dz)/reps

        if self.boundary: # Convert to Blakelet
        
            # Image regularised stokeslet
            dz = x[:, 2] + y[2]
            
            r2 = dx*dx+dy*dy+dz*dz
            reps = (r2 + eps2)**1.5
            reps_p = r2 + 2*eps2
            
            Gxx = Gxx - (reps_p + dx*dx)/reps
            Gxy = Gxy - dx*dy/reps
            Gxz = Gxz - dx*dz/reps
            Gyy = Gyy - (reps_p + dy*dy)/reps
            Gyz = Gyz - dy*dz/reps
            Gzz = Gzz - (reps_p + dz*dz)/reps
            
            Gyx = Gxy
            Gzx = Gxz
            Gzy = Gyz
            
            # Blob term
            reps2 = (r2 + eps2)**2.5
            h = y[2]
            h2 = h*h
            blob = -6*eps2*h2/reps2
            
            Gxx = Gxx + blob
            Gyy = Gyy + blob
            Gzz = Gzz - blob
            
            # Rotlet difference term
            rot_term = -6*h*eps2/reps2
            
            Gxx = Gxx - dz*rot_term
            Gyy = Gyy - dz*rot_term
            
            Gzx = Gzx + dx*rot_term
            Gzy = Gzy + dy*rot_term
            
            # Potential source dipole
            Gxx = Gxx + 2*h2*(1/reps - 3*dx*dx/reps2)
            Gxy = Gxy + 2*h2*(-3*dx*dy/reps2)
            Gxz = Gxz - 2*h2*(-3*dx*dz/reps2)
            
            Gyx = Gyx + 2*h2*(-3*dy*dx/reps2)
            Gyy = Gyy + 2*h2*(1/reps - 3*dy*dy/reps2)
            Gyz = Gyz - 2*h2*(-3*dy*dz/reps2)
            
            Gzx = Gzx + 2*h2*(-3*dz*dx/reps2)
            Gzy = Gzy + 2*h2*(-3*dz*dy/reps2)
            Gzz = Gzz - 2*h2*(1./reps - 3*dz*dz/reps2)
            
            # Stokes dipole
            reps_f = r2 + eps2
            prefac = 2*h/reps2
            Gxx = Gxx + prefac*(- dz*reps_f + 3*dx*dx*dz)
            Gxy = Gxy + prefac*(3*dx*dy*dz)
            Gxz = Gxz - prefac*(- dx*reps_f + 3*dx*dz*dz)
            
            Gyx = Gyx + prefac*(3*dy*dx*dz)
            Gyy = Gyy + prefac*(- dz*reps_f + 3*dy*dy*dz)
            Gyz = Gyz - prefac*(- dy*reps_f + 3*dy*dz*dz)
            
            Gzx = Gzx + prefac*(dx*(r2 + 4*eps2) + 3*dz*dx*dz)
            Gzy = Gzy + prefac*(dy*(r2 + 4*eps2) + 3*dz*dy*dz)
            Gzz = Gzz - prefac*(dz*(r2 + 4*eps2) - 2*dz*reps_f + 3*dz*dz*dz)
        
        G = np.stack([
            np.stack([Gxx, Gxy, Gxz], axis=-1),
            np.stack([Gxy, Gyy, Gyz], axis=-1),
            np.stack([Gxz, Gyz, Gzz], axis=-1),
        ], axis=-2)
        if x.shape[0] == 1:
            return G[0]
        return G
    
    def fluid_flow(self, x: np.ndarray, t: float):
        """
        Returns the current fluid flow at position x and time t.

        Parameters
        ----------
        x: ndarray
            Coordinates in space to evaluate.
        t: float
            Time to evaluate.

        Returns
        -------
        u: ndarray
            Velocities of the flow at (x, t). 
        """

        x_flat = np.stack((x[0].ravel(), x[1].ravel(), x[2].ravel()), axis=-1)
        u = np.zeros((x_flat.shape[0], self.dim))
        for dumbbell in range(self.num_of_dumbbells):
            xr = self.bead_pos[2*dumbbell+1] - self.bead_pos[2*dumbbell]
            distance = np.sqrt((xr*xr).sum())
            spring_force = self.k*(distance-self.rest_length)/(self.mu*self.rest_length*distance)*xr
            gravity_force = np.zeros(self.dim)
            gravity_force[2] = -self.gravity
            repulsion_force_1 = np.zeros(self.dim)
            repulsion_force_2 = np.zeros(self.dim)
            for bead in range(self.num_of_beads):
                xr_1 = self.bead_pos[bead] - self.bead_pos[2*dumbbell]
                dists_m_1 = np.sqrt(
                    (xr_1*xr_1).sum()
                )
                xr_2 = self.bead_pos[bead] - self.bead_pos[2*dumbbell+1]
                dists_m_2 = np.sqrt(
                    (xr_2*xr_2).sum()
                )
                repulsion_force_1 += -(self.repulsion*np.exp(-dists_m_1**2/self.repulsion_range)*np.nan_to_num(xr_1/dists_m_1))
                repulsion_force_2 += -(self.repulsion*np.exp(-dists_m_2**2/self.repulsion_range)*np.nan_to_num(xr_2/dists_m_2))
            s1 = np.einsum('ijk,k->ij', self.regularised_stokeslet(x_flat, self.bead_pos[2*dumbbell]), spring_force + gravity_force + repulsion_force_1)
            s2 = np.einsum('ijk,k->ij', self.regularised_stokeslet(x_flat, self.bead_pos[2*dumbbell+1]), -spring_force + gravity_force + repulsion_force_2)
            u += (s1 + s2)/(8*np.pi*self.mu)
        
        u += self.bg_flow(x_flat, t)

        return u.reshape(x[0].shape + (self.dim,))
    
    def bead_dynamics(self, t, positions):
        """
        Creates ODE system wrapper for SciPy's solve_ivp routine.
        """
        positions = positions.reshape(self.num_of_beads, self.dim)
        u_array = np.zeros((self.num_of_beads, self.dim))
        for n in range(self.num_of_beads):
            x = positions[n]
            u = np.zeros(3)
            for dumbbell in range(self.num_of_dumbbells):
                xr = positions[2*dumbbell+1] - positions[2*dumbbell]
                distance = np.sqrt((xr*xr).sum())
                spring_force = self.k*(distance-self.rest_length)/(self.mu*self.rest_length*distance)*xr
                gravity_force = np.zeros(self.dim)
                gravity_force[2] = -self.gravity
                repulsion_force_1 = np.zeros(self.dim)
                repulsion_force_2 = np.zeros(self.dim)
                for bead in range(self.num_of_beads):
                    xr_1 = positions[bead] - positions[2*dumbbell]
                    dists_m_1 = np.sqrt(
                        (xr_1*xr_1).sum()
                    )
                    xr_2 = positions[bead] - positions[2*dumbbell+1]
                    dists_m_2 = np.sqrt(
                        (xr_2*xr_2).sum()
                    )
                    repulsion_force_1 += -(self.repulsion*np.exp(-dists_m_1**2/self.repulsion_range)*np.nan_to_num(xr_1/dists_m_1))
                    repulsion_force_2 += -(self.repulsion*np.exp(-dists_m_2**2/self.repulsion_range)*np.nan_to_num(xr_2/dists_m_2))
                s1 = self.regularised_stokeslet(x, positions[2*dumbbell]) @ (spring_force + gravity_force + repulsion_force_1)
                s2 = self.regularised_stokeslet(x, positions[2*dumbbell+1]) @ (-spring_force + gravity_force + repulsion_force_2)
                u += (s1 + s2)/(8*np.pi*self.mu)
            
            u += self.bg_flow(x, t)
            u_array[n] = u

        return np.ravel(u_array)
    
    def solve_dynamics(self, max_time: Optional[float] = 1.0,
                       method: Optional[str] = 'RK45', 
                       verbose: Optional[bool] = False) -> None:
        """
        Solves beads ODE system using SciPy's solve_ivp

        Parameters
        ----------
        max_time: float, optional
            Maximum time of simulation, default is 1.0.
        method: str, optional
            Method to pass to solve_ivp, default is RK45.
        verbose: bool, optional
            Set to True for more information during simulation.
        """

        if verbose:
            start = time.time()
            print('Solving bead dynamics... \n')
        self.bead_sol = solve_ivp(self.bead_dynamics, 
                                  [0, max_time], 
                                  y0 = np.ravel(self.bead_pos), 
                                  dense_output=True, 
                                  method=method)

        if verbose:
            end = time.time()
            print(f'Solved bead dynamics, took {end-start:2f} seconds.')

    def create_3d_animation(self, domain: Optional[list] = None, 
                         grid_points: Optional[int] = 10, 
                         n_timesteps: Optional[int] = 100,
                         arrow_size: Optional[float] = 1.0,
                         same_size: Optional[bool] = False, 
                         filename: Optional[str] = None) -> None:
        """
        Creates three-dimensional animation of beads and fluid flow.

        Parameters
        ----------
        domain: list, optional 
            The space domain to visualise, default is None.
        grid_points: int, optional
            Number of evenly spaced grid points used to calculate the flow, default is 10.
        n_timesteps: int, optional
            Number of evenly spaced timesteps to calculate the evolution, default is 100.
        arrow_size: float, optional
            Size of the quiver arrows, default is 1.0.
        same_size: bool, optional
            Set to true to have all arrows have the same length.
        filename: str, optional
            None to not create a video file, 
            otherwise string will be used as filename (extension and video format not included). 
        """

        if self.bead_sol is None:
            raise ValueError('Solve bead dynamics first')
        
        if domain is None:
            domain = [[-1, -1, -1], [1, 1, 1]]

        x_flat = np.linspace(domain[0][0], domain[1][0], grid_points)
        y_flat = np.linspace(domain[0][1], domain[1][1], grid_points)
        z_flat = np.linspace(domain[0][2], domain[1][2], grid_points)
        X = np.meshgrid(x_flat, y_flat, z_flat)
        U = self.fluid_flow(X, 0)
        u, v, w = U[..., 0], U[..., 1], U[..., 2]
        x, y, z = X

        f = plt.figure(figsize=(6, 5))
        ax = f.add_subplot(111, projection='3d')
        ax.set_xlim(domain[0][0], domain[1][0])
        ax.set_ylim(domain[0][1], domain[1][1])
        ax.set_zlim(domain[0][2], domain[1][2])
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'$z$')
        ax.set_title(r'Time: 0.0')

        self.quiver = ax.quiver(x, y, z, u, v, w, color='r', length=arrow_size, normalize=same_size)

        lines = [None]*self.num_of_dumbbells
        for d in range(self.num_of_dumbbells):
            lines[d], = ax.plot(self.bead_pos[2*d:2*d+2][:, 0], self.bead_pos[2*d:2*d+2][:, 1],
                                zs=self.bead_pos[2*d:2*d+2][:, 2],
                                 marker='o', lw=4)
        max_t = self.bead_sol.t[-1]
        t = np.linspace(0, max_t, n_timesteps)
        sol = self.bead_sol.sol(t)
        def update(fn):
            ax.set_title(fr'Time: {t[fn]:2f}')
            self.bead_pos = sol[:, fn].reshape(self.num_of_beads, 3)
            U = self.fluid_flow(X, t[fn])
            u, v, w = U[..., 0], U[..., 1], U[..., 2]
            
            for d in range(self.num_of_dumbbells):
                lines[d].set_data_3d(self.bead_pos[2*d:2*d+2][:, 0], self.bead_pos[2*d:2*d+2][:, 1], self.bead_pos[2*d:2*d+2][:, 2])
            self.quiver.remove()
            self.quiver = ax.quiver(x, y, z, u, v, w, color='r', length=arrow_size, normalize=same_size)

        animation = FuncAnimation(f, update, interval=1, frames=n_timesteps)
        if filename is not None:
            animation.save(f'./videos/{filename}', writer='ffmpeg', fps=20)
        plt.show()
    
    def create_2d_projection(self, domain: Optional[list] = None, 
                             grid_points: Optional[int] = 100, 
                             n_timesteps: Optional[int] = 100, 
                             filename: Optional[str] = None) -> None:
        """
        Creates a two-dimensional projection plot, by removing the x-axis.

        Parameters
        ----------
        domain: list, optional
            The space domain to evaluate, default is None. x-axis will be projected.
        """

        if self.bead_sol is None:
            raise ValueError('Solve bead dynamics first')
        
        if domain is None:
            domain = [[-1, -1, -1], [1, 1, 1]]
        
        x_flat = np.linspace(domain[0][0], domain[1][0], grid_points)
        y_flat = np.linspace(domain[0][1], domain[1][1], grid_points)
        z_flat = np.linspace(domain[0][2], domain[1][2], grid_points)
        mp = len(x_flat) // 2
        X = np.meshgrid(x_flat, y_flat, z_flat)
        U = self.fluid_flow(X, 0)
        u, v, w = U[..., 0], U[..., 1], U[..., 2]
        x, y, z = X
        proj_y, proj_z = np.meshgrid(y_flat, z_flat)
    

        f = plt.figure(figsize=(6, 5))
        ax = f.add_subplot(111)
        # X is Y, Y is Z
        ax.set_xlim(domain[0][1], domain[1][1])
        ax.set_ylim(domain[0][2], domain[1][2])
        ax.set_xlabel(r'$y$')
        ax.set_ylabel(r'$z$')
        ax.set_title(r'Time: 0.0')
        
        #self.stream = ax.streamplot(proj_y, proj_z, proj_v, proj_w, color='red')
        self.heatmap = ax.pcolormesh(proj_y, proj_z, np.sqrt(u.mean(axis=0)*u.mean(axis=0)+v.mean(axis=0)*v.mean(axis=0)+w.mean(axis=0)*w.mean(axis=0)), alpha=1, cmap='PiYG',
                                      norm=colors.LogNorm(vmin=0.001, vmax=100))
        lines = [None]*self.num_of_dumbbells
        for d in range(self.num_of_dumbbells):
            lines[d], = ax.plot(self.bead_pos[2*d:2*d+2][:, 1], self.bead_pos[2*d:2*d+2][:, 2],
                                 marker='o', lw=4)
        f.colorbar(self.heatmap, label=r'$|u| = \sqrt{u_i u_i}$')
        max_t = self.bead_sol.t[-1]
        t = np.linspace(0, max_t, n_timesteps)
        sol = self.bead_sol.sol(t)
        def update(fn):
            ax.set_title(fr'Time: {t[fn]:2f}')
            self.bead_pos = sol[:, fn].reshape(self.num_of_beads, 3)
            U = self.fluid_flow(X, t[fn])
            u, v, w = U[..., 0], U[..., 1], U[..., 2]
            h = np.sqrt(u.mean(axis=0)*u.mean(axis=0) + v.mean(axis=0)*v.mean(axis=0) + w.mean(axis=0)*w.mean(axis=0))
            print(h.max(), h.min())
            self.heatmap.set_array(h)
            
            for d in range(self.num_of_dumbbells):
                lines[d].set_data(self.bead_pos[2*d:2*d+2][:, 1], self.bead_pos[2*d:2*d+2][:, 2])
            #self.stream.lines.remove()
            #for art in ax.get_children():
            #    if not isinstance(art, mpl.patches.FancyArrowPatch):
            #        continue
            #    art.remove()
            
            #self.stream = ax.streamplot(proj_y, proj_z, v[mp, :, :], w[mp, :, :], color='red')
        animation = FuncAnimation(f, update, interval=1, frames=n_timesteps)
        if filename is not None:
            animation.save(f'./videos/{filename}', writer='ffmpeg', fps=20)
        plt.show()

class FlowLibrary:

    def __init__(self, shear: Optional[float] = 1.0,
                 strain: Optional[float] = 1.0,
                 ang_freq: Optional[float] = 1.0):
        """
        Constructor for the FlowLibrary, includes a collection of standard background flows.

        Parameters
        ----------
        shear: float, optional
            Shear strength, default is 1.0.
        strain: float, optional
            Strain strength, default is 1.0.
        ang_freq: float, optional
            Angular frequency, default is 1.0.
        """
        self.shear = shear
        self.strain = strain
        self.ang_freq = ang_freq
    
    def shear_flow_3d(self, x, t) -> np.ndarray:
        return np.stack([np.zeros_like(x[..., 1]),
                         self.shear*x[..., 2], 
                         np.zeros_like(x[..., 1])], axis=-1)
    
    def osc_shear_flow_3d(self, x, t) -> np.ndarray:
        return np.stack([np.zeros_like(x[..., 1]),
                         self.shear*np.cos(self.ang_freq*t)*x[..., 2],
                         np.zeros_like(x[..., 1])], axis=-1)
    
    def extensional_flow_3d(self, x, t) -> np.ndarray:
        return np.stack([
                        np.zeros_like(x[..., 0]),
                         self.strain*x[..., 1], 
                         -self.strain*x[..., 2],
                         ], axis=-1)
