from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize, root_scalar
import math
import os
from typing import Dict, List
import multiprocessing as mp
from functools import partial
import time
#from combined_plot_spin import plot_spin_polarization
#from freeenergy_magnatization_test import one_state
# Color and style settings
colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
styles = {"FD": "-", "FLB": "--", "NFL": "-."}
labels = {"FD": "FD", "FLB": "FLB", "NFL": "NFL"}

# Dispersion for the tight-binding model:
def tight_binding_dispersion(kx, ky, t, tp, a=1.0):
    return 2 * t * (np.cos(kx*a) + np.cos(ky*a)) + 2 * tp * (np.cos((kx+ky)*a) + np.cos((kx-ky)*a))

# Free-energy functional per state for FD (for a given spin)
def free_energy_FD(n, epsilon, T):
    # n is the occupation, 0<n<1
    # We use the conventional FD free energy (up to an additive constant)
    return epsilon * n + T * (n * np.log(n) + (1 - n) * np.log(1 - n))

# For a given k-point, with energies e_up and e_down, our total free energy is:
def func_FD(x, e_up, e_down, T):
    # x[0] is n_up, x[1] is n_down.
    eps_val = 1e-7
    n_up = np.clip(x[0], eps_val, 1 - eps_val)
    n_down = np.clip(x[1], eps_val, 1 - eps_val)
    F_up = free_energy_FD(n_up, e_up, T)
    F_down = free_energy_FD(n_down, e_down, T)
    return F_up + F_down

# Its derivative with respect to n (Jacobian):
def func_deriv_FD(x, e_up, e_down, T):
    eps_val = 1e-8
    n_up = np.clip(x[0], eps_val, 1 - eps_val)
    n_down = np.clip(x[1], eps_val, 1 - eps_val)
    # Derivative: dF/dn = ε + T*(ln n - ln(1-n))
    dF_dnup = e_up + T * (np.log(n_up) - np.log(1 - n_up))
    dF_dndown = e_down + T * (np.log(n_down) - np.log(1 - n_down))
    return np.array([dF_dnup, dF_dndown], dtype=float)

# We no longer need heavy constraints because the analytic minimum ensures 0<n<1,
# but to be safe we can impose simple bounds.
bounds_FD = [(1e-7, 1 - 1e-7), (1e-7, 1 - 2e-7)]
eps = 1e-10
cons_FD = (
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[0]-eps]),
     'jac' : lambda x: np.array([1.0, 0.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[0]]),
     'jac' : lambda x: np.array([-1.0, 0.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[1]-eps]),
     'jac' : lambda x: np.array([0.0, 1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[1]]),
     'jac' : lambda x: np.array([0.0, -1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[0]+x[1]-eps]),
     'jac' : lambda x: np.array([1.0, 1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([2-eps-x[0]-x[1]]),
     'jac' : lambda x: np.array([-1.0, -1.0])}
)
# For a given kx, ky, define e_up and e_down for the FD case:

def nk_FD(kx, ky, T, h, mu, t, tp, a=1.0, n_target=None):
    # Define energies: adjust signs to match your FD formula.
    # Here we follow the convention of your analytic FD in the first code snippet.
    e_up = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) - h/2 - mu
    e_down = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) + h/2 - mu
    
    initial_guess = [n_target/2, n_target/2] 
    #print("n_init=",initial_guess,end="") 
    # Use SLSQP minimization for free energy:
    res = minimize(func_FD, initial_guess, args=(e_up, e_down, T),jac=func_deriv_FD,
                   constraints=cons_FD, method='SLSQP', options={'eps':1e-10, 'disp': False})
    if not res.success:
        print("Minimize did not converge at k=({:.2f},{:.2f}): {}".format(kx, ky, res.message))
        return np.array([0.0, 0.0])
    else:
        return res.x

"""
# For a given kx, ky, define e_up and e_down for the FD case:
def nk_FD(kx, ky, T, h, mu, t, tp, a=1.0):
    # Define energies: adjust signs to match your FD formula.
    # Here we follow the convention of your analytic FD in the first code snippet.
    e_up = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) - h/2 - mu
    e_down = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) + h/2 - mu
    # Use SLSQP minimization for free energy:
    res = minimize(func_FD, [0.4, 0.4], args=(e_up, e_down, T),jac=func_deriv_FD,
                   bounds=bounds_FD, method='SLSQP', options={'eps':1e-10, 'disp': False})
    if not res.success:
        print("Minimize did not converge at k=({:.2f},{:.2f}): {}".format(kx, ky, res.message))
        return np.array([0.0, 0.0])
    else:
        return res.x
"""
def n_tot_FD(T, h, mu, L, t, tp, a=1.0, n_target=None):
    result_up = 0.0
    result_down = 0.0
    for i in range(L):
        kx = i * 2 * np.pi / L
        for j in range(L):
            ky = j * 2 * np.pi / L
            sol = nk_FD(kx, ky, T, h, mu, t, tp, a, n_target)
            result_up += sol[0]
            result_down += sol[1]
    # Average over the grid (and note spin degeneracy is already included here as we solve separately for up and down)
    n_total_up = result_up / (L**2)
    n_total_down = result_down / (L**2)
    return n_total_up, n_total_down

def n_tot_FD_parallel(T, h, mu, L, t, tp, a=1.0, n_cores=None, n_target=None):
    """Parallelized version of n_tot_FD using multiprocessing."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Create k-point pairs with all parameters
    k_points = []
    for i in range(L):
        kx = i * 2 * np.pi / L
        for j in range(L):
            ky = j * 2 * np.pi / L
            k_points.append((kx, ky, T, h, mu, t, tp, a, n_target))
    
    # Process in parallel using global worker function
    with mp.Pool(n_cores) as pool:
        results = pool.map(worker_FD, k_points)
    
    # Sum results
    result_up = sum(r[0] for r in results)
    result_down = sum(r[1] for r in results)
    
    n_total_up = result_up / (L**2)
    n_total_down = result_down / (L**2)
    return n_total_up, n_total_down

def density_FD(mu, T, h, L, n_target, t, tp, a=1.0):
    n_up, n_down = n_tot_FD(T, h, mu, L, t, tp, a, n_target)
    return n_up + n_down - n_target

def density_FD_parallel(mu, T, h, L, n_target, t, tp, a=1.0, n_cores=None):
    """Parallelized version of density_FD."""
    n_up, n_down = n_tot_FD_parallel(T, h, mu, L, t, tp, a, n_cores, n_target)
    return n_up + n_down - n_target

# Chemical potential solver for FD using brentq (bracketing is robust for monotonic FD):
def solve_mu_FD(n_target, T, L, h, t, tp, a=1.0, mu_low=-80, mu_high=80):
    return root_scalar(lambda mu: density_FD(mu, T, h, L, n_target, t, tp, a),
                       bracket=[mu_low, mu_high], method='brentq').root

def solve_mu_FD_parallel(n_target, T, L, h, t, tp, a=1.0, mu_low=-80, mu_high=80, n_cores=None):
    """Parallelized version of solve_mu_FD."""
    return root_scalar(lambda mu: density_FD_parallel(mu, T, h, L, n_target, t, tp, a, n_cores),
                       bracket=[mu_low, mu_high], method='brentq').root

cons_NFL = (
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[0]-eps]),
     'jac' : lambda x: np.array([1.0, 0.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[0]]),
     'jac' : lambda x: np.array([-1.0, 0.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[1]-eps]),
     'jac' : lambda x: np.array([0.0, 1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[1]]),
     'jac' : lambda x: np.array([0.0, -1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[0]+x[1]-eps]),
     'jac' : lambda x: np.array([1.0, 1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[0]-x[1]]),
     'jac' : lambda x: np.array([-1.0, -1.0])}
)
cons_FLB = (
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[0]-eps]),
     'jac' : lambda x: np.array([1.0, 0.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[0]]),
     'jac' : lambda x: np.array([-1.0, 0.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[1]-eps]),
     'jac' : lambda x: np.array([0.0, 1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([1-eps-x[1]]),
     'jac' : lambda x: np.array([0.0, -1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([x[0]+x[1]-eps]),
     'jac' : lambda x: np.array([1.0, 1.0])},
    {'type': 'ineq',
     'fun' : lambda x: np.array([2-eps-x[0]-x[1]]),
     'jac' : lambda x: np.array([-1.0, -1.0])}
)
def func_NFL(x, e_up, e_down, T):
    n_up = x[0]
    n_down = x[1]
    return -T*(-2*n_down*np.log(n_down) - 2*n_up*np.log(n_up) + (1 - n_down)*np.log(1 - n_down) + (1 - n_up)*np.log(1 - n_up) + 
               (n_down + n_up)*np.log(n_down + n_up) - (-2*n_down - 2*n_up + 2)*np.log(-n_down - n_up + 1)) + e_down*n_down + e_up*n_up

def func_deriv_NFL(x, e_up, e_down, T):
    n_up = x[0]
    n_down = x[1]
    return np.array([-T*(-2*np.log(n_up) - np.log(1 - n_up) + np.log(n_down + n_up) + 2*np.log(-n_down - n_up + 1) - 2 - (2*n_down + 2*n_up - 2)/(-n_down - n_up + 1)) + e_up, 
                     -T*(-2*np.log(n_down) - np.log(1 - n_down) + np.log(n_down + n_up) + 2*np.log(-n_down - n_up + 1) - 2 - (2*n_down + 2*n_up - 2)/(-n_down - n_up + 1)) + e_down])

def nk_NFL(kx, ky, T, h, mu, t, tp, a=1.0, n_target=None):
    e_up = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) - h/2 - mu
    e_down = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) + h/2 - mu
    
    initial_guess = [n_target/2, n_target/2] 
    
    res = minimize(func_NFL, initial_guess, args=(e_up, e_down, T), jac=func_deriv_NFL,
       constraints=cons_NFL, method='SLSQP', options={'eps':1e-10, 'disp': False})
    if not res.success:
        print("Minimize did not converge at k=({:.2f},{:.2f}): {}".format(kx, ky, res.message))
        sol = [0.0, 0.0]
    else:
        sol = res.x
    return res.x

def n_tot_NFL(T, h, mu, L, t, tp, a=1.0, n_target=None):
    result = 0
    for i in range(0, L):
        kx = i*2*math.pi/L
        for j in range(0, L):
            ky = j*2*math.pi/L
            result += nk_NFL(kx, ky, T, h, mu, t, tp, a, n_target)
    return result/L/L

def n_tot_NFL_parallel(T, h, mu, L, t, tp, a=1.0, n_cores=None, n_target=None):
    """Parallelized version of n_tot_NFL using multiprocessing."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Create k-point pairs with all parameters
    k_points = []
    for i in range(L):
        kx = i * 2 * np.pi / L
        for j in range(L):
            ky = j * 2 * np.pi / L
            k_points.append((kx, ky, T, h, mu, t, tp, a, n_target))
    
    # Process in parallel using global worker function
    with mp.Pool(n_cores) as pool:
        results = pool.map(worker_NFL, k_points)
    
    # Sum results
    result = sum(results)
    return result / (L**2)

def density_NFL(mu, T, h, L, n_target, t, tp, a=1.0):
    n = n_tot_NFL(T, h, mu, L, t, tp, a, n_target)
    return n[0] + n[1] - n_target

def density_NFL_parallel(mu, T, h, L, n_target, t, tp, a=1.0, n_cores=None):
    """Parallelized version of density_NFL."""
    n = n_tot_NFL_parallel(T, h, mu, L, t, tp, a, n_cores, n_target)
    return n[0] + n[1] - n_target

def solve_mu_NFL(n_target, T, L, h, t, tp, a=1.0, mu_low=-80, mu_high=80):
    return root_scalar(lambda mu: density_NFL(mu, T, h, L, n_target, t, tp, a),
                       bracket=[mu_low, mu_high], method='brentq').root

def solve_mu_NFL_parallel(n_target, T, L, h, t, tp, a=1.0, mu_low=-80, mu_high=80, n_cores=None):
    """Parallelized version of solve_mu_NFL."""
    return root_scalar(lambda mu: density_NFL_parallel(mu, T, h, L, n_target, t, tp, a, n_cores),
                       bracket=[mu_low, mu_high], method='brentq').root

def func_FLB(x, e_up, e_down, T):
    # Use a slightly larger epsilon for safety
    eps_val = 1e-6
    n_up = np.clip(x[0], eps_val, 1 - eps_val)
    n_down = np.clip(x[1], eps_val, 1 - eps_val)
    F = e_up * n_up + e_down * n_down - T * (
        - 2*n_up*np.log(n_up) - 2*n_down*np.log(n_down) - (1-n_up)*np.log(1-n_up) - (1-n_down)*np.log(1-n_down)
        + (n_up+n_down)*np.log(n_up+n_down) 
    )
    return F

def func_deriv_FLB(x, e_up, e_down, T):
    n_up = x[0]
    n_down = x[1]
    return np.array([- T*(np.log(n_down + n_up) - 2*np.log(n_up) + np.log(-n_up + 1)) + e_up,- T*(np.log(n_down + n_up) - 2*np.log(n_down) + np.log(-n_down + 1)) + e_down
])

def nk_FLB(kx, ky, T, h, mu, t, tp, a=1.0, n_target=None):
    e_up = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) - h/2 - mu
    e_down = -2.0*(np.cos(kx) + np.cos(ky)) + 4.0*tp*np.cos(kx)*np.cos(ky) + h/2 - mu
    
    initial_guess = [n_target/2, n_target/2] 
    
    res = minimize(func_FLB, initial_guess, args=(e_up, e_down, T), jac=func_deriv_FLB,
       constraints=cons_FLB, method='SLSQP', options={'eps':1e-10, 'disp': False})
    if not res.success:
        #print("Minimize did not converge at k=({:.2f},{:.2f}): {}".format(kx, ky, res.message))
        sol = [0.0, 0.0]
    else:
        sol = res.x
    return res.x


def n_tot_FLB(T, h, mu, L, t, tp, a=1.0, n_target=None):
    result = 0
    for i in range(0, L):
        kx = i*2*math.pi/L
        for j in range(0, L):
            ky = j*2*math.pi/L
            result += nk_FLB(kx, ky, T, h, mu, t, tp, a, n_target)
    return result/L/L

def n_tot_FLB_parallel(T, h, mu, L, t, tp, a=1.0, n_cores=None, n_target=None):
    """Parallelized version of n_tot_FLB using multiprocessing."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Create k-point pairs with all parameters
    k_points = []
    for i in range(L):
        kx = i * 2 * np.pi / L
        for j in range(L):
            ky = j * 2 * np.pi / L
            k_points.append((kx, ky, T, h, mu, t, tp, a, n_target))
    
    # Process in parallel using global worker function
    with mp.Pool(n_cores) as pool:
        results = pool.map(worker_FLB, k_points)
    
    # Sum results
    result = sum(results)
    return result / (L**2)

def density_FLB(mu, T, h, L, n_target, t, tp, a=1.0):
    n = n_tot_FLB(T, h, mu, L, t, tp, a, n_target)
    return n[0] + n[1] - n_target

def density_FLB_parallel(mu, T, h, L, n_target, t, tp, a=1.0, n_cores=None):
    """Parallelized version of density_FLB."""
    n = n_tot_FLB_parallel(T, h, mu, L, t, tp, a, n_cores, n_target)
    return n[0] + n[1] - n_target

def solve_mu_FLB(n_target, T, L, h, t, tp, a=1.0, mu_low=-80, mu_high=80):
    return root_scalar(lambda mu: density_FLB(mu, T, h, L, n_target, t, tp, a),
                       bracket=[mu_low, mu_high], method='brentq').root

def solve_mu_FLB_parallel(n_target, T, L, h, t, tp, a=1.0, mu_low=-80, mu_high=80, n_cores=None):
    """Parallelized version of solve_mu_FLB."""
    return root_scalar(lambda mu: density_FLB_parallel(mu, T, h, L, n_target, t, tp, a, n_cores),
                       bracket=[mu_low, mu_high], method='brentq').root

def one_state(dist: str, T: float, n_target: float, h: float, kx: np.ndarray, ky: np.ndarray, t: float, tp: float, a: float) -> Dict:
    """Return a dict row for the results DataFrame."""
    # use grid length as integer L
    L = len(kx)
    if dist == "FD":
        mu = solve_mu_FD(n_target, T, L, h, t, tp, a)
        n_up, n_dn = n_tot_FD(T, h, mu, L, t, tp, a, n_target)
    elif dist == "FLB":
        mu = solve_mu_FLB(n_target, T, L, h, t, tp, a)
        n_up, n_dn = n_tot_FLB(T, h, mu, L, t, tp, a, n_target)
    elif dist == "NFL":
        mu = solve_mu_NFL(n_target, T, L, h, t, tp, a)
        n_up, n_dn = n_tot_NFL(T, h, mu, L, t, tp, a, n_target)
    else:
        raise ValueError(dist)

    return {
        "f": dist,
        "T": T,
        "$n_{target}$": n_target,
        "h": h,
        "mu": mu,
        "n_up": n_up,
        "n_down": n_dn,
    }

def one_state_parallel(dist: str, T: float, n_target: float, h: float, kx: np.ndarray, ky: np.ndarray, t: float, tp: float, a: float, n_cores: int = None) -> Dict:
    """Parallelized version of one_state function."""
    # use grid length as integer L
    L = len(kx)
    if dist == "FD":
        mu = solve_mu_FD_parallel(n_target, T, L, h, t, tp, a, n_cores=n_cores)
        n_up, n_dn = n_tot_FD_parallel(T, h, mu, L, t, tp, a, n_cores, n_target)
    elif dist == "FLB":
        mu = solve_mu_FLB_parallel(n_target, T, L, h, t, tp, a, n_cores=n_cores)
        n_up, n_dn = n_tot_FLB_parallel(T, h, mu, L, t, tp, a, n_cores, n_target)
    elif dist == "NFL":
        mu = solve_mu_NFL_parallel(n_target, T, L, h, t, tp, a, n_cores=n_cores)
        n_up, n_dn = n_tot_NFL_parallel(T, h, mu, L, t, tp, a, n_cores, n_target)
    else:
        raise ValueError(dist)

    return {
        "f": dist,
        "T": T,
        "$n_{target}$": n_target,
        "h": h,
        "mu": mu,
        "n_up": n_up,
        "n_down": n_dn,
    }

temps = [0.01,0.1,0.5,1.0]
n_target_list= list(np.linspace(0.1,0.9,9))
n_target_list.insert(0,0.01)
n_target_list.append(0.99) # different target densities

# Example usage (commented out - see main block below)
# T = 0.01
# tp = 0.25
# h = 0.0
# temps=[0.01]
# n_target_list=[0.8]
# L = 1
# n_target = 0.8
# t = -1.0
# a = 1.0

# for T in temps:
#     for n in n_target_list:
#         mu_fd = solve_mu_FD(n, T, L, h, t, tp, a)
#         print("T={},n={},FD chemical potential (entropy minimization method: SLSQP): {}".format(T, n, mu_fd))
# for T in temps:
#     for n in n_target_list:
#         mu_nfl = solve_mu_NFL(n, T, L, h, t, tp, a)
#         print("T={},n={},NFL chemical potential (entropy minimization method: SLSQP): {}".format(T, n, mu_nfl))
# for T in temps:
#     for n in n_target_list:
#         mu_flb = solve_mu_FLB(n, T, L, h, t, tp, a)
#         print("T={},n={},FLB chemical potential (entropy minimization method: SLSQP): {}".format(T, n, mu_flb))



def plot_spin_polarization(results_df, output_dir, distributions, t, tp):
    """Plot spin polarization (n_up - n_down) vs temperature and target density."""
    
    # Calculate spin polarization
    results_df['polarization'] = results_df['n_up'] - results_df['n_down']
    results_df['polarization_ratio'] = (results_df['n_up'] - results_df['n_down']) / (results_df['n_up'] + results_df['n_down'])
    
    # Get unique n_target and temperature values
    n_target_values = sorted(results_df['$n_{target}$'].astype(float).unique())
    temperature_values = sorted(results_df['T'].unique())
    
    # Plot polarization vs temperature for each n_target
    for n_target in n_target_values:
        # Filter data for this n_target
        n_df = results_df[results_df['$n_{target}$'].astype(float) == n_target]
        
        plt.figure(figsize=(10, 6))
        
        # Color and style settings
        colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
        styles = {"FD": "-", "FLB": "--", "NFL": "-."}
        
        for dist in distributions:
            dist_df = n_df[n_df['f'] == dist]
            
            if not dist_df.empty:
                # Sort by temperature
                dist_df = dist_df.sort_values(by='T')
                
                plt.plot(
                    dist_df['T'],
                    dist_df['polarization'],
                    color=colors.get(dist, "black"),
                    linestyle=styles.get(dist, "-"),
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    label=f'{dist}'
                )
        
        plt.xlabel(r'Temperature $T$', fontsize=16, labelpad=10)
        plt.ylabel(r'Magnetic moment, $(n_{\uparrow} - n_{\downarrow})$', fontsize=16, labelpad=10)
        
        # Add horizontal line at n_target for reference
        plt.axhline(y=n_target, color='blue', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'$n = {n_target:.2f}$')
        
        # Add legend with professional styling
        legend = plt.legend(fontsize=14, loc='upper left', frameon=False)
        
        # Professional tick styling
        plt.tick_params(
            axis='both',
            which='both',
            direction='in',
            bottom=True, top=True,
            left=True, right=True,
            labelsize=15,
            width=3,
            length=8,
            pad=6
        )
        
        # Scientific notation formatting
        from matplotlib.ticker import ScalarFormatter
        fmt = ScalarFormatter(useMathText=True, useOffset=False)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0,0))
        plt.gca().xaxis.set_major_formatter(fmt)
        
        # Add temperature annotation (no background box for professional look)
        temp_value = dist_df['T'].iloc[0] if not dist_df.empty else temperature_values[0] if temperature_values else 0.001
        temp_sci = format_temperature_scientific(temp_value)
        plt.annotate(f'$\\frac{{k_B T}}{{|t|}} = {temp_sci}$', 
                    xy=(0.05, 0.75), xytext=(0.05, 0.75),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=16, ha='left', va='top')
        
        # No grid for cleaner professional look
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(output_dir, f"SpinPol_vs_T_n{n_target}_T{temp_value}_{'_'.join(distributions)}_t{t}_tp{tp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Spin Polarization vs Temperature plot for n={n_target} saved as {filename}")
    
    # Plot polarization vs n_target for each temperature
    for temperature in temperature_values:
        # Filter data for this temperature
        temp_df = results_df[results_df['T'] == temperature]
        
        plt.figure(figsize=(10, 6))
        
        # Color and style settings
        colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
        styles = {"FD": "-", "FLB": "--", "NFL": "-."}
        
        for dist in distributions:
            dist_df = temp_df[temp_df['f'] == dist]
            
            if not dist_df.empty:
                # Convert n_target to float and sort
                dist_df['n_target_float'] = dist_df['$n_{target}$'].astype(float)
                dist_df = dist_df.sort_values(by='n_target_float')
                
                plt.plot(
                    dist_df['n_target_float'],
                    dist_df['polarization'],
                    color=colors.get(dist, "black"),
                    linestyle=styles.get(dist, "-"),
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    label=f'{dist}'
                )
        
        plt.xlabel(r'Target Density $n$', fontsize=16, labelpad=10)
        plt.ylabel(r'Magnetic moment, $(n_{\uparrow} - n_{\downarrow})$', fontsize=16, labelpad=10)
        plt.title(f'Spin Polarization vs Target Density for T={temperature:.5f}, t={t}, t\'={tp}', fontsize=14)
        
        # Add legend with professional styling
        legend = plt.legend(fontsize=14, loc='upper left', frameon=False)
        
        # Professional tick styling
        plt.tick_params(
            axis='both',
            which='both',
            direction='in',
            bottom=True, top=True,
            left=True, right=True,
            labelsize=15,
            width=3,
            length=8,
            pad=6
        )
        
        # Scientific notation formatting
        from matplotlib.ticker import ScalarFormatter
        fmt = ScalarFormatter(useMathText=True, useOffset=False)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0,0))
        plt.gca().xaxis.set_major_formatter(fmt)
        
        # Add temperature annotation (no background box for professional look)
        temp_sci = format_temperature_scientific(temperature)
        plt.annotate(f'$\\frac{{k_B T}}{{|t|}} = {temp_sci}$', 
                    xy=(0.05, 0.75), xytext=(0.05, 0.75),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=16, ha='left', va='top')
        
        # No grid for cleaner professional look
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(output_dir, f"SpinPol_vs_n_T{temperature}_{'_'.join(distributions)}_t{t}_tp{tp}.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Spin Polarization vs Target Density plot for T={temperature} saved as {filename}")

def run_simulation(
    temperatures: List[float],
    n_target_list: List[float],
    h_values: np.ndarray,
    distributions: List[str] = ("FD", "FLB", "NFL"),
    #distributions: List[str] = ["FLB"],
    Nk: int = 100,
    t: float = -1.0,
    tp: float = 0.25,
    a: float = 1.0,
    out_root: str = "plots_spin_tb_all_freeenergy_FD_FLB_1n_NFL_1n_test",
    plot_distributions: bool = True,
    n_target_per_dist: dict = None,  # New: allows different n_target for each distribution
):
    """High‑level sweep over T, n_target and Zeeman field h.
    
    Parameters:
    - n_target_list: List of target densities (used when n_target_per_dist is None)
    - n_target_per_dist: Dict like {"FD": [0.8, 1.6], "FLB": [0.8, 1.6], "NFL": [0.4, 0.8]}
                         If provided, n_target_list is ignored and each distribution uses its own targets
    """

    kx = np.linspace(0, 2 * np.pi / a, Nk, endpoint=False)
    ky = kx.copy()

    # Determine which parameter set to use
    if n_target_per_dist is not None:
        print(f"Using DISTRIBUTION-SPECIFIC n_targets:")
        for dist, targets in n_target_per_dist.items():
            print(f"  {dist}: {targets}")
        master_out = f"{out_root}_per_dist_t{t}_tp{tp}"
        use_per_dist = True
    else:
        print(f"Using SAME n_targets for all distributions: {n_target_list}")
        master_out = f"{out_root}_t{t}_tp{tp}"
        use_per_dist = False
    
    os.makedirs(master_out, exist_ok=True)

    global_rows = []
    mag_vs_h = {dist: [] for dist in distributions}

    for h in h_values:
        print(f"\n=== h = {h:.9e} ===")
        rows_h = []
        for T in temperatures:
            if use_per_dist:
                # Use distribution-specific n_targets
                for dist in distributions:
                    if dist in n_target_per_dist:
                        for n_target in n_target_per_dist[dist]:
                            row = one_state(dist, T, n_target=n_target, h=h, kx=kx, ky=ky, t=t, tp=tp, a=a)
                            rows_h.append(row)
                            global_rows.append(row)
                
                # Generate distribution plots for first temperature and first h value
                if plot_distributions and T == temperatures[0] and h == h_values[0]:
                    # Use first n_target for each distribution for plotting
                    first_targets = {dist: targets[0] for dist, targets in n_target_per_dist.items() if targets}
                    dist_plot_dir = os.path.join(master_out, f"distributions_T{T}_per_dist")
                    plot_distributions_per_dist(T, h, first_targets, Nk, t, tp, a, distributions, dist_plot_dir)
            else:
                # Traditional mode: same n_target for all distributions
                for n_target in n_target_list:
                    for dist in distributions:
                        row = one_state(dist, T, n_target=n_target, h=h, kx=kx, ky=ky, t=t, tp=tp, a=a)
                        rows_h.append(row)
                        global_rows.append(row)
                
                # Generate distribution plots for first temperature and first h value
                if plot_distributions and T == temperatures[0] and h == h_values[0]:
                    for n_target in n_target_list[:1]:  # Just first n_target for plotting
                        dist_plot_dir = os.path.join(master_out, f"distributions_T{T}_n{n_target}")
                        plot_all_distributions(T, h, n_target, Nk, t, tp, a, distributions, dist_plot_dir)
                    
        # ---- per‑h plots -------------------------------------------------- #
        h_folder = os.path.join(master_out, f"h_{h:.10f}")
        os.makedirs(h_folder, exist_ok=True)
        df_h = pd.DataFrame(rows_h)
        plot_spin_polarization(df_h, h_folder, distributions, t, tp)
        #plot_entropy_vs_temperature(df_h, h_folder, distributions, t, tp)
        #plot_entropy_vs_density(df_h, h_folder, distributions, t, tp)

        # store mean M for aggregated M(h)
        for dist in distributions:
            seg = df_h[df_h["f"] == dist]
            M_avg = (seg["n_up"] - seg["n_down"]).mean() if not seg.empty else np.nan
            mag_vs_h[dist].append(M_avg)
            print(f"  {dist:<3}: ⟨M⟩ = {M_avg:.4f}")

    # ---- aggregated M(h) --------------------------------------------------- #
    plt.figure(figsize=(12, 8))
    for dist in distributions:
        plt.plot(h_values * 1e4, mag_vs_h[dist], styles[dist], marker="o", color=colors[dist], 
                label=labels[dist], linewidth=3, markersize=8)
    
    plt.xlabel(r"Zeeman field, $h/|t| \times 10^{4}$", fontsize=16, labelpad=10)
    plt.ylabel(r"Magnetic moment, $(n_{\uparrow} - n_{\downarrow})$", fontsize=16, labelpad=10)
    title_suffix = "per-distribution targets" if use_per_dist else "same targets"
    #plt.title(f"Average magnetization vs h ({title_suffix})")
    
    # Add reference line for n_target
    if use_per_dist and n_target_per_dist:
        # For per-dist mode, show first n_target as reference
        first_target = list(n_target_per_dist.values())[0][0] if n_target_per_dist else 0.8
        plt.axhline(y=first_target, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    else:
        # For same-target mode, show the actual n_target
        ref_target = n_target_list[0] if n_target_list else 0.8
        plt.axhline(y=ref_target, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add legend with professional styling
    legend = plt.legend(fontsize=14, loc='upper left', frameon=False)
    
    # Professional tick styling
    plt.tick_params(
        axis='both',
        which='both',
        direction='in',
        bottom=True, top=True,
        left=True, right=True,
        labelsize=15,
        width=2,
        length=8,
        pad=6
    )
    
    # Manual scaling - x-axis values should be scaled by 1e4 for the label to be correct
    # Since label says "× 10^4", the actual plotted values should be h*1e4
    # No automatic scientific notation formatting since we handle it manually in the label
    
    # Set reasonable number of ticks
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    # Add temperature annotation (no background box for professional look)
    temp_value = temperatures[0] if temperatures else 0.001
    temp_sci = format_temperature_scientific(temp_value)
    plt.annotate(f'$\\frac{{k_B T}}{{|t|}} = {temp_sci}$', 
                xy=(0.05, 0.95), xytext=(0.05, 0.75),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=16, ha='left', va='top')
    
    # Add n_target annotation
    if use_per_dist and n_target_per_dist:
        target_info = ", ".join([f"{dist}:{targets[0]:.1f}" for dist, targets in n_target_per_dist.items() if targets])
        plt.annotate(f'$n$ targets: {target_info}', 
                    xy=(0.95, 0.25), xytext=(0.95, 0.25),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=14, ha='right', va='bottom')
    else:
        n_target_str = ', '.join([f'{n:.1f}' for n in n_target_list]) if n_target_list else 'N/A'
        plt.annotate(f'$n = {n_target_str}$', 
                    xy=(0.95, 0.25), xytext=(0.95, 0.25),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=16, ha='right', va='bottom')
    
    # No grid for cleaner professional look
    plt.tight_layout()
    plt.savefig(os.path.join(master_out, f"Aggregated_M_vs_h_T{temperatures[0]}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save raw data ---------------------------------------------------------- #
    df_all = pd.DataFrame(global_rows)
    df_all.to_csv(os.path.join(master_out, "results_raw.csv"), index=False)
    print("\nSimulation finished. Raw data stored in results_raw.csv")

def run_simulation_parallel(
    temperatures: List[float],
    n_target_list: List[float],
    h_values: np.ndarray,
    distributions: List[str] = ("FD", "FLB", "NFL"),
    Nk: int = 100,
    t: float = -1.0,
    tp: float = 0.25,
    a: float = 1.0,
    out_root: str = "plots_spin_tb_all_freeenergy_FD_FLB_1n_NFL_1n_test_parallel",
    plot_distributions: bool = True,
    n_target_per_dist: dict = None,
    n_cores: int = None,
):
    """
    Parallelized version of run_simulation.
    
    Parameters:
    - n_cores: Number of CPU cores to use (None for auto-detection)
    """
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    print(f"Running parallelized simulation with {n_cores} CPU cores")
    start_time = time.time()

    kx = np.linspace(0, 2 * np.pi / a, Nk, endpoint=False)
    ky = kx.copy()

    # Determine which parameter set to use
    if n_target_per_dist is not None:
        print(f"Using DISTRIBUTION-SPECIFIC n_targets:")
        for dist, targets in n_target_per_dist.items():
            print(f"  {dist}: {targets}")
        master_out = f"{out_root}_per_dist_t{t}_tp{tp}"
        use_per_dist = True
    else:
        print(f"Using SAME n_targets for all distributions: {n_target_list}")
        master_out = f"{out_root}_t{t}_tp{tp}"
        use_per_dist = False
    
    os.makedirs(master_out, exist_ok=True)

    global_rows = []
    mag_vs_h = {dist: [] for dist in distributions}

    for h in h_values:
        print(f"\n=== h = {h:.9e} ===")
        rows_h = []
        for T in temperatures:
            if use_per_dist:
                # Use distribution-specific n_targets
                for dist in distributions:
                    if dist in n_target_per_dist:
                        for n_target in n_target_per_dist[dist]:
                            row = one_state_parallel(dist, T, n_target=n_target, h=h, kx=kx, ky=ky, t=t, tp=tp, a=a, n_cores=n_cores)
                            rows_h.append(row)
                            global_rows.append(row)
                
                # Generate distribution plots for first temperature and first h value
                if plot_distributions and T == temperatures[0] and h == h_values[0]:
                    # Use first n_target for each distribution for plotting
                    first_targets = {dist: targets[0] for dist, targets in n_target_per_dist.items() if targets}
                    dist_plot_dir = os.path.join(master_out, f"distributions_T{T}_per_dist")
                    plot_distributions_per_dist(T, h, first_targets, Nk, t, tp, a, distributions, dist_plot_dir)
            else:
                # Traditional mode: same n_target for all distributions
                for n_target in n_target_list:
                    for dist in distributions:
                        row = one_state_parallel(dist, T, n_target=n_target, h=h, kx=kx, ky=ky, t=t, tp=tp, a=a, n_cores=n_cores)
                        rows_h.append(row)
                        global_rows.append(row)
                
                # Generate distribution plots for first temperature and first h value
                if plot_distributions and T == temperatures[0] and h == h_values[0]:
                    for n_target in n_target_list[:1]:  # Just first n_target for plotting
                        dist_plot_dir = os.path.join(master_out, f"distributions_T{T}_n{n_target}")
                        plot_all_distributions(T, h, n_target, Nk, t, tp, a, distributions, dist_plot_dir)
                    
        # ---- per‑h plots -------------------------------------------------- #
        h_folder = os.path.join(master_out, f"h_{h:.10f}")
        os.makedirs(h_folder, exist_ok=True)
        df_h = pd.DataFrame(rows_h)
        plot_spin_polarization(df_h, h_folder, distributions, t, tp)

        # store mean M for aggregated M(h)
        for dist in distributions:
            seg = df_h[df_h["f"] == dist]
            M_avg = (seg["n_up"] - seg["n_down"]).mean() if not seg.empty else np.nan
            mag_vs_h[dist].append(M_avg)
            print(f"  {dist:<3}: ⟨M⟩ = {M_avg:.4f}")

    # ---- aggregated M(h) --------------------------------------------------- #
    plt.figure(figsize=(12, 8))
    for dist in distributions:
        plt.plot(h_values * 1e4, mag_vs_h[dist], styles[dist], marker="o", color=colors[dist], 
                label=labels[dist], linewidth=3, markersize=8)
    
    plt.xlabel(r"Zeeman field, $h/|t| \times 10^{4}$", fontsize=16, labelpad=10)
    plt.ylabel(r"Magnetic moment, $(n_{\uparrow} - n_{\downarrow})$", fontsize=16, labelpad=10)
    title_suffix = "per-distribution targets" if use_per_dist else "same targets"
    
    # Add reference line for n_target
    if use_per_dist and n_target_per_dist:
        first_target = list(n_target_per_dist.values())[0][0] if n_target_per_dist else 0.8
        plt.axhline(y=first_target, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    else:
        ref_target = n_target_list[0] if n_target_list else 0.8
        plt.axhline(y=ref_target, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add legend with professional styling
    legend = plt.legend(fontsize=14, loc='upper left', frameon=False)
    
    # Professional tick styling
    plt.tick_params(
        axis='both',
        which='both',
        direction='in',
        bottom=True, top=True,
        left=True, right=True,
        labelsize=15,
        width=2,
        length=8,
        pad=6
    )
    
    # Set reasonable number of ticks
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    
    # Add temperature annotation
    temp_value = temperatures[0] if temperatures else 0.001
    temp_sci = format_temperature_scientific(temp_value)
    plt.annotate(f'$\\frac{{k_B T}}{{|t|}} = {temp_sci}$', 
                xy=(0.05, 0.95), xytext=(0.05, 0.75),
                xycoords='axes fraction', textcoords='axes fraction',
                fontsize=16, ha='left', va='top')
    
    # Add n_target annotation
    if use_per_dist and n_target_per_dist:
        target_info = ", ".join([f"{dist}:{targets[0]:.1f}" for dist, targets in n_target_per_dist.items() if targets])
        plt.annotate(f'$n$ targets: {target_info}', 
                    xy=(0.95, 0.25), xytext=(0.95, 0.25),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=14, ha='right', va='bottom')
    else:
        n_target_str = ', '.join([f'{n:.1f}' for n in n_target_list]) if n_target_list else 'N/A'
        plt.annotate(f'$n = {n_target_str}$', 
                    xy=(0.95, 0.25), xytext=(0.95, 0.25),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=16, ha='right', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(master_out, f"Aggregated_M_vs_h_T{temperatures[0]}_parallel.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Save raw data
    df_all = pd.DataFrame(global_rows)
    df_all.to_csv(os.path.join(master_out, "results_raw_parallel.csv"), index=False)
    
    end_time = time.time()
    print(f"\nParallel simulation finished in {end_time - start_time:.2f} seconds")
    print(f"Raw data stored in results_raw_parallel.csv")

###############################################################################
# 7.  Command‑line entry point                                               #
###############################################################################

def test_distribution_plots():
    """
    Example function to test the new distribution plotting capabilities.
    Run this to generate sample distribution plots.
    """
    # Parameters for testing
    T = 0.01
    h = 0.0
    n_target = 0.8
    L = 20  # Smaller grid for faster testing
    t = -1.0
    tp = 0.25
    a = 1.0
    distributions = ["FD", "FLB", "NFL"]
    
    # Create test output directory
    test_output = "test_distribution_plots"
    os.makedirs(test_output, exist_ok=True)
    
    print("Generating test distribution plots...")
    print(f"Parameters: T={T}, h={h}, n_target={n_target}, L={L}x{L}")
    
    # Generate comprehensive distribution plots
    plot_all_distributions(T, h, n_target, L, t, tp, a, distributions, test_output)
    
    print(f"\nTest plots saved in '{test_output}' directory")
    print("Generated plots include:")
    print("  - Individual distribution plots in subdirectories (FD_plots/, FLB_plots/, NFL_plots/)")
    print("  - Comparison plots in comparison_plots/ subdirectory")
    print("  - Each includes: n_k vs energy, n_k vs momentum, and 2D heatmaps")

def compute_distribution_data(T, h, mu, L, t, tp, a, distributions, n_target=None):
    """
    Compute k-resolved occupation data for all distributions at given mu.
    Returns dictionary with energy and occupation arrays for plotting.
    """
    kx = np.linspace(0, 2 * np.pi / a, L, endpoint=False)
    ky = kx.copy()
    
    # Compute energies for all k-points
    energies = []
    kx_flat = []
    ky_flat = []
    
    for i in range(L):
        for j in range(L):
            kx_val = kx[i]
            ky_val = ky[j]
            energy = tight_binding_dispersion(kx_val, ky_val, t, tp, a)
            energies.append(energy)
            kx_flat.append(kx_val)
            ky_flat.append(ky_val)
    
    energies = np.array(energies)
    kx_flat = np.array(kx_flat)
    ky_flat = np.array(ky_flat)
    
    # Compute occupations for each distribution
    occupation_data = {}
    
    for dist in distributions:
        n_up_array = []
        n_down_array = []
        n_total_array = []
        
        for i in range(L):
            for j in range(L):
                kx_val = kx[i]
                ky_val = ky[j]
                
                if dist == "FD":
                    sol = nk_FD(kx_val, ky_val, T, h, mu, t, tp, a, n_target)
                elif dist == "FLB":
                    sol = nk_FLB(kx_val, ky_val, T, h, mu, t, tp, a, n_target)
                elif dist == "NFL":
                    sol = nk_NFL(kx_val, ky_val, T, h, mu, t, tp, a, n_target)
                else:
                    sol = [0.0, 0.0]
                
                n_up_array.append(sol[0])
                n_down_array.append(sol[1])
                n_total_array.append(sol[0] + sol[1])
        
        occupation_data[dist] = {
            'n_up': np.array(n_up_array),
            'n_down': np.array(n_down_array),
            'n_total': np.array(n_total_array)
        }
    
    return {
        'energies': energies,
        'kx': kx_flat,
        'ky': ky_flat,
        'occupations': occupation_data
    }

def plot_distribution_vs_energy(data_dict, output_dir, distributions, T, h, mu, t, tp, n_target):
    """
    Plot occupation numbers vs energy for different distributions.
    
    Parameters:
    - data_dict: Output from compute_distribution_data()
    - output_dir: Directory to save plots
    - distributions: List of distributions to plot
    - T, h, mu, t, tp, n_target: Physical parameters for labeling
    """
    energies = data_dict['energies']
    occupations = data_dict['occupations']
    
    # Color and style settings
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    
    # Sort by energy for cleaner plots
    sort_indices = np.argsort(energies)
    energies_sorted = energies[sort_indices]
    
    # Plot total occupation (n_up + n_down)
    plt.figure(figsize=(12, 8))
    
    for dist in distributions:
        if dist in occupations:
            n_total_sorted = occupations[dist]['n_total'][sort_indices]
            plt.plot(energies_sorted, n_total_sorted,
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=2, marker='o', markersize=3, 
                    label=f'{dist}', alpha=0.8)
    
    plt.xlabel('Energy $\\epsilon_k/|t|$', fontsize=14)
    plt.ylabel('Total Occupation $n_k^\\uparrow + n_k^\\downarrow$', fontsize=14)
    plt.title(f'Occupation vs Energy\\nT={T:.4f}, h={h:.6f}, μ={mu:.3f}, n={n_target:.2f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"nk_vs_energy_total_T{T}_h{h:.6f}_n{n_target}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Total occupation vs energy plot saved as {filename}")
    
    # Plot spin-resolved occupations side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spin-up occupations
    for dist in distributions:
        if dist in occupations:
            n_up_sorted = occupations[dist]['n_up'][sort_indices]
            ax1.plot(energies_sorted, n_up_sorted,
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=2, marker='o', markersize=3,
                    label=f'{dist}', alpha=0.8)
    
    ax1.set_xlabel('Energy $\\epsilon_k/|t|$', fontsize=12)
    ax1.set_ylabel('Spin-Up Occupation $n_k^\\uparrow$', fontsize=12)
    ax1.set_title(f'Spin-Up Occupation vs Energy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Spin-down occupations
    for dist in distributions:
        if dist in occupations:
            n_down_sorted = occupations[dist]['n_down'][sort_indices]
            ax2.plot(energies_sorted, n_down_sorted,
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=2, marker='o', markersize=3,
                    label=f'{dist}', alpha=0.8)
    
    ax2.set_xlabel('Energy $\\epsilon_k/|t|$', fontsize=12)
    ax2.set_ylabel('Spin-Down Occupation $n_k^\\downarrow$', fontsize=12)
    ax2.set_title(f'Spin-Down Occupation vs Energy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'Spin-Resolved Occupations: T={T:.4f}, h={h:.6f}, μ={mu:.3f}, n={n_target:.2f}', fontsize=16)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"nk_vs_energy_spin_T{T}_h{h:.6f}_n{n_target}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spin-resolved occupation vs energy plot saved as {filename}")

def plot_distribution_vs_momentum(data_dict, output_dir, distributions, T, h, mu, t, tp, n_target):
    """
    Plot occupation numbers along high-symmetry lines in k-space.
    """
    kx = data_dict['kx']
    ky = data_dict['ky']
    occupations = data_dict['occupations']
    
    # Color and style settings
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    
    # Extract data along kx=0 line (vertical cut through BZ)
    kx_line_indices = np.where(np.abs(kx) < 0.1)[0]  # Points near kx=0
    ky_line = ky[kx_line_indices]
    
    # Sort by ky for plotting
    sort_indices = np.argsort(ky_line)
    ky_line_sorted = ky_line[sort_indices]
    line_indices_sorted = kx_line_indices[sort_indices]
    
    plt.figure(figsize=(12, 8))
    
    for dist in distributions:
        if dist in occupations:
            n_total_line = occupations[dist]['n_total'][line_indices_sorted]
            plt.plot(ky_line_sorted, n_total_line,
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=3, marker='o', markersize=6,
                    label=f'{dist}')
    
    plt.xlabel('Momentum $k_y$', fontsize=14)
    plt.ylabel('Total Occupation $n_k^\\uparrow + n_k^\\downarrow$', fontsize=14)
    plt.title(f'Occupation vs Momentum (k_x≈0)\\nT={T:.4f}, h={h:.6f}, μ={mu:.3f}, n={n_target:.2f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"nk_vs_ky_T{T}_h{h:.6f}_n{n_target}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Occupation vs momentum plot saved as {filename}")

def plot_distribution_heatmap(data_dict, output_dir, distributions, T, h, mu, t, tp, n_target, L):
    """
    Plot 2D heatmaps of occupation in k-space for each distribution.
    """
    kx = data_dict['kx'].reshape(L, L)
    ky = data_dict['ky'].reshape(L, L)
    occupations = data_dict['occupations']
    
    colors = {"FD": "Blues", "FLB": "Reds", "NFL": "Greens"}
    
    fig, axes = plt.subplots(1, len(distributions), figsize=(5*len(distributions), 4))
    if len(distributions) == 1:
        axes = [axes]
    
    for i, dist in enumerate(distributions):
        if dist in occupations:
            n_total_2d = occupations[dist]['n_total'].reshape(L, L)
            
            im = axes[i].imshow(n_total_2d, extent=[0, 2*np.pi, 0, 2*np.pi], 
                              origin='lower', cmap=colors[dist], aspect='equal')
            axes[i].set_xlabel('$k_x$', fontsize=12)
            axes[i].set_ylabel('$k_y$', fontsize=12)
            axes[i].set_title(f'{dist}', fontsize=14)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('$n_k^\\uparrow + n_k^\\downarrow$', fontsize=10)
    
    fig.suptitle(f'Occupation Heatmaps: T={T:.4f}, h={h:.6f}, μ={mu:.3f}, n={n_target:.2f}', fontsize=16)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"nk_heatmap_T{T}_h{h:.6f}_n{n_target}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Occupation heatmap saved as {filename}")

def plot_all_distributions(T, h, n_target, L, t, tp, a, distributions, output_dir):
    """
    Generate comprehensive distribution plots for given parameters.
    """
    # First solve for mu to achieve target density (using FD as reference)
    mu_fd = solve_mu_FD(n_target, T, L, h, t, tp, a)
    
    # For comparison, also compute mu for other distributions
    mu_values = {"FD": mu_fd}
    if "FLB" in distributions:
        mu_values["FLB"] = solve_mu_FLB(n_target, T, L, h, t, tp, a)
    if "NFL" in distributions:
        mu_values["NFL"] = solve_mu_NFL(n_target, T, L, h, t, tp, a)
    
    print(f"\nChemical potentials for n_target={n_target}, T={T}, h={h}:")
    for dist, mu in mu_values.items():
        print(f"  {dist}: μ = {mu:.6f}")
    
    # Generate plots using each distribution's self-consistent mu
    for dist in distributions:
        if dist in mu_values:
            mu = mu_values[dist]
            
            # Create subdirectory for this distribution
            dist_dir = os.path.join(output_dir, f"{dist}_plots")
            os.makedirs(dist_dir, exist_ok=True)
            
            # Compute distribution data
            data = compute_distribution_data(T, h, mu, L, t, tp, a, [dist], n_target)
            
            # Generate all plot types
            plot_distribution_vs_energy(data, dist_dir, [dist], T, h, mu, t, tp, n_target)
            plot_distribution_vs_momentum(data, dist_dir, [dist], T, h, mu, t, tp, n_target)
            plot_distribution_heatmap(data, dist_dir, [dist], T, h, mu, t, tp, n_target, L)
    
    # Also create comparison plots using FD chemical potential for all distributions
    comparison_dir = os.path.join(output_dir, "comparison_plots")
    os.makedirs(comparison_dir, exist_ok=True)
    
    data_comparison = compute_distribution_data(T, h, mu_fd, L, t, tp, a, distributions, n_target)
    plot_distribution_vs_energy(data_comparison, comparison_dir, distributions, T, h, mu_fd, t, tp, n_target)
    plot_distribution_vs_momentum(data_comparison, comparison_dir, distributions, T, h, mu_fd, t, tp, n_target)

def plot_distributions_per_dist(T, h, n_target_dict, L, t, tp, a, distributions, output_dir):
    """
    Generate distribution plots where each distribution has its own n_target.
    
    Parameters:
    - n_target_dict: Dict like {"FD": 0.8, "FLB": 0.8, "NFL": 0.4}
    """
    print(f"\n{'='*60}")
    print(f"PER-DISTRIBUTION PLOTS")
    print(f"{'='*60}")
    
    # Create a dummy kx array for the one_state function
    kx = np.linspace(0, 2 * np.pi / a, L, endpoint=False)
    ky = kx.copy()
    
    # Compute chemical potentials for each distribution
    mu_values = {}
    
    for dist in distributions:
        if dist in n_target_dict:
            n_target = n_target_dict[dist]
            result = one_state(dist, T, n_target=n_target, h=h, kx=kx, ky=ky, t=t, tp=tp, a=a)
            mu_values[dist] = result['mu']
            print(f"{dist}: μ = {mu_values[dist]:.6f}, n_target = {n_target:.3f}")
    
    # Generate plots for each distribution at their respective chemical potentials
    for dist in distributions:
        if dist in mu_values and dist in n_target_dict:
            mu = mu_values[dist]
            n_target = n_target_dict[dist]
            
            # Create subdirectory for this distribution
            dist_dir = os.path.join(output_dir, f"{dist}_plots_n{n_target:.3f}")
            os.makedirs(dist_dir, exist_ok=True)
            
            # Compute distribution data
            data = compute_distribution_data(T, h, mu, L, t, tp, a, [dist], n_target)
            
            # Generate all plot types
            plot_distribution_vs_energy(data, dist_dir, [dist], T, h, mu, t, tp, n_target)
            plot_distribution_vs_momentum(data, dist_dir, [dist], T, h, mu, t, tp, n_target)
            plot_distribution_heatmap(data, dist_dir, [dist], T, h, mu, t, tp, n_target, L)
    
    # Create comparison plots
    comparison_dir = os.path.join(output_dir, "comparison_per_dist")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compute all distributions at their respective chemical potentials for comparison
    all_data = {}
    for dist in distributions:
        if dist in mu_values:
            mu = mu_values[dist]
            all_data[dist] = compute_distribution_data(T, h, mu, L, t, tp, a, [dist], n_target)
    
    # Create a combined comparison plot
    if len(all_data) > 1:
        plot_per_dist_comparison(all_data, comparison_dir, distributions, T, h, mu_values, n_target_dict, t, tp)
    
    print(f"\nPer-distribution plots saved in: {output_dir}")

def plot_per_dist_comparison(all_data, output_dir, distributions, T, h, mu_values, n_target_dict, t, tp):
    """Create a comparison plot for per-distribution targets."""
    
    plt.figure(figsize=(14, 10))
    
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    
    for dist in distributions:
        if dist in all_data:
            data = all_data[dist]
            energies = data['energies']
            occupations = data['occupations'][dist]
            
            sort_indices = np.argsort(energies)
            energies_sorted = energies[sort_indices]
            n_total_sorted = occupations['n_total'][sort_indices]
            
            n_target = n_target_dict[dist]
            mu = mu_values[dist]
            
            plt.plot(energies_sorted, n_total_sorted,
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=3, marker='o', markersize=3,
                    label=f'{dist} (n={n_target:.3f}, μ={mu:.3f})',
                    alpha=0.8)
    
    plt.xlabel('Energy $\\epsilon_k/|t|$', fontsize=16)
    plt.ylabel('Total Occupation $n_k^\\uparrow + n_k^\\downarrow$', fontsize=16)
    plt.title(f'Per-Distribution Comparison\\n'
              f'T={T:.4f}, h={h:.6f}, t={t}, t\'={tp}', fontsize=18)
    
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"per_dist_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-distribution comparison plot saved: {filename}")

###############################################################################
# ENTROPY CALCULATION FUNCTIONS
###############################################################################

def entropy_k_point(n_up, n_down, T, distribution):
    """
    Calculate the entropy contribution from a single k-point.
    
    Parameters:
    - n_up, n_down: Spin-up and spin-down occupations
    - T: Temperature
    - distribution: Distribution type ("FD", "FLB", "NFL")
    
    Returns:
    - Entropy contribution from this k-point
    """
    # Ensure valid ranges to avoid NaNs
    eps = 1e-12
    n_up = np.clip(n_up, eps, 1.0 - eps)
    n_down = np.clip(n_down, eps, 1.0 - eps)
    n_total = n_up + n_down
    
    if distribution == "FD":
        # Standard Fermi-Dirac entropy
        s_up = -n_up * np.log(n_up) - (1 - n_up) * np.log(1 - n_up)
        s_down = -n_down * np.log(n_down) - (1 - n_down) * np.log(1 - n_down)
        return s_up + s_down
    
    elif distribution == "FLB":
        # Fermi Liquid Boltzmann entropy
        if n_total > eps and n_up > eps and n_down > eps:
            return (-2*n_up*np.log(n_up) - 2*n_down*np.log(n_down) -
                    (1-n_up)*np.log(1-n_up) - (1-n_down)*np.log(1-n_down) +
                    n_total*np.log(n_total))
        else:
            return 0.0
    
    elif distribution == "NFL":
        # Non-Fermi Liquid entropy with exclusion constraint
        if n_total > eps and n_up > eps and n_down > eps and n_total < (1.0 - eps):
            return (-2*n_up*np.log(n_up) - 2*n_down*np.log(n_down) +
                    (1-n_up)*np.log(1-n_up) + (1-n_down)*np.log(1-n_down) +
                    n_total*np.log(n_total) -
                    (2-2*n_total)*np.log(1-n_total))
        else:
            return 0.0
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def compute_total_entropy(T, h, mu, L, t, tp, a, distribution, n_target=None):
    """
    Compute the total entropy integrated over all k-points.
    
    Parameters:
    - T, h, mu: Temperature, magnetic field, chemical potential
    - L: Grid size
    - t, tp, a: Tight-binding parameters
    - distribution: Distribution type
    - n_target: Target density for initial guess (optional)
    
    Returns:
    - Total entropy
    """
    kx = np.linspace(0, 2 * np.pi / a, L, endpoint=False)
    ky = kx.copy()
    
    total_entropy = 0.0
    
    for i in range(L):
        for j in range(L):
            kx_val = kx[i]
            ky_val = ky[j]
            
            # Get occupations at this k-point
            if distribution == "FD":
                sol = nk_FD(kx_val, ky_val, T, h, mu, t, tp, a, n_target)
            elif distribution == "FLB":
                sol = nk_FLB(kx_val, ky_val, T, h, mu, t, tp, a, n_target)
            elif distribution == "NFL":
                sol = nk_NFL(kx_val, ky_val, T, h, mu, t, tp, a, n_target)
            else:
                sol = [0.0, 0.0]
            
            n_up, n_down = sol[0], sol[1]
            
            # Add entropy contribution from this k-point
            entropy_k = entropy_k_point(n_up, n_down, T, distribution)
            total_entropy += entropy_k
    
    # Apply k-space integration weight
    weight = (2*np.pi / L)**2 / (2*np.pi)**2
    
    return total_entropy * weight

def compute_total_entropy_parallel(T, h, mu, L, t, tp, a, distribution, n_cores=None, n_target=None):
    """
    Parallelized version of compute_total_entropy.
    """
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    # Create k-point pairs with all parameters
    k_points = []
    for i in range(L):
        kx = i * 2 * np.pi / L
        for j in range(L):
            ky = j * 2 * np.pi / L
            k_points.append((kx, ky, T, h, mu, t, tp, a, distribution, n_target))
    
    # Process in parallel using global worker function
    with mp.Pool(n_cores) as pool:
        entropy_contributions = pool.map(worker_entropy, k_points)
    
    # Sum results
    total_entropy = sum(entropy_contributions)
    
    # Apply k-space integration weight
    weight = (2*np.pi / L)**2 / (2*np.pi)**2
    
    return total_entropy * weight

###############################################################################
# ENHANCED PLOTTING FUNCTIONS
###############################################################################

def plot_chemical_potential_vs_temperature(results_df, output_dir, distributions, t, tp):
    """Plot chemical potential vs temperature for each distribution and target density."""
    
    print(f"\n{'='*60}")
    print(f"PLOTTING CHEMICAL POTENTIAL VS TEMPERATURE")
    print(f"{'='*60}")
    
    # Get unique n_target values
    n_target_values = sorted(results_df['$n_{target}$'].unique())
    
    # Color and style settings
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    labels = {"FD": "Fermi-Dirac", "FLB": "Fermi Liquid", "NFL": "Non-Fermi Liquid"}
    
    # Create a separate plot for each n_target
    for n_target in n_target_values:
        n_df = results_df[results_df['$n_{target}$'] == n_target]
        
        plt.figure(figsize=(10, 6))
        
        for dist in distributions:
            dist_df = n_df[n_df['f'] == dist]
            
            if not dist_df.empty:
                # Sort by temperature
                dist_df = dist_df.sort_values(by='T')
                
                plt.plot(
                    dist_df['T'],
                    dist_df['mu'],
                    color=colors[dist],
                    linestyle=styles[dist],
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    label=labels[dist]
                )
        
        plt.xlabel('Temperature $T/|t|$', fontsize=14)
        plt.ylabel('Chemical Potential $\\mu/|t|$', fontsize=14)
        plt.title(f'Chemical Potential vs Temperature\\nn = {n_target:.3f}, t = {t}, t\' = {tp}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"mu_vs_T_n{n_target:.3f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chemical potential vs temperature plot saved: {filename}")

def plot_chemical_potential_vs_density(results_df, output_dir, distributions, t, tp):
    """Plot chemical potential vs target density for each distribution and temperature."""
    
    print(f"\n{'='*60}")
    print(f"PLOTTING CHEMICAL POTENTIAL VS DENSITY")
    print(f"{'='*60}")
    
    # Get unique temperature values
    temperature_values = sorted(results_df['T'].unique())
    
    # Color and style settings
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    labels = {"FD": "Fermi-Dirac", "FLB": "Fermi Liquid", "NFL": "Non-Fermi Liquid"}
    
    # Create a separate plot for each temperature
    for temperature in temperature_values:
        temp_df = results_df[results_df['T'] == temperature]
        
        plt.figure(figsize=(10, 6))
        
        for dist in distributions:
            dist_df = temp_df[temp_df['f'] == dist]
            
            if not dist_df.empty:
                # Sort by n_target
                dist_df = dist_df.sort_values(by='$n_{target}$')
                
                plt.plot(
                    dist_df['$n_{target}$'],
                    dist_df['mu'],
                    color=colors[dist],
                    linestyle=styles[dist],
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    label=labels[dist]
                )
        
        plt.xlabel('Target Density $n$', fontsize=14)
        plt.ylabel('Chemical Potential $\\mu/|t|$', fontsize=14)
        plt.title(f'Chemical Potential vs Density\\nT = {temperature:.4f}|t|, t = {t}, t\' = {tp}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"mu_vs_n_T{temperature:.4f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Chemical potential vs density plot saved: {filename}")

def plot_entropy_vs_temperature(results_df, output_dir, distributions, t, tp):
    """Plot entropy vs temperature for each distribution and target density."""
    
    print(f"\n{'='*60}")
    print(f"PLOTTING ENTROPY VS TEMPERATURE")
    print(f"{'='*60}")
    
    # Get unique n_target values
    n_target_values = sorted(results_df['$n_{target}$'].unique())
    
    # Color and style settings
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    labels = {"FD": "Fermi-Dirac", "FLB": "Fermi Liquid", "NFL": "Non-Fermi Liquid"}
    
    # Create a separate plot for each n_target
    for n_target in n_target_values:
        n_df = results_df[results_df['$n_{target}$'] == n_target]
        
        plt.figure(figsize=(10, 6))
        
        for dist in distributions:
            dist_df = n_df[n_df['f'] == dist]
            
            if not dist_df.empty:
                # Sort by temperature
                dist_df = dist_df.sort_values(by='T')
                
                plt.plot(
                    dist_df['T'],
                    dist_df['S'],
                    color=colors[dist],
                    linestyle=styles[dist],
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    label=labels[dist]
                )
        
        plt.xlabel('Temperature $T/|t|$', fontsize=14)
        plt.ylabel('Entropy $S/k_B$', fontsize=14)
        plt.title(f'Entropy vs Temperature\\nn = {n_target:.3f}, t = {t}, t\' = {tp}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"S_vs_T_n{n_target:.3f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Entropy vs temperature plot saved: {filename}")

def plot_entropy_vs_density(results_df, output_dir, distributions, t, tp):
    """Plot entropy vs target density for each distribution and temperature."""
    
    print(f"\n{'='*60}")
    print(f"PLOTTING ENTROPY VS DENSITY")
    print(f"{'='*60}")
    
    # Get unique temperature values
    temperature_values = sorted(results_df['T'].unique())
    
    # Color and style settings
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    labels = {"FD": "Fermi-Dirac", "FLB": "Fermi Liquid", "NFL": "Non-Fermi Liquid"}
    
    # Create a separate plot for each temperature
    for temperature in temperature_values:
        temp_df = results_df[results_df['T'] == temperature]
        
        plt.figure(figsize=(10, 6))
        
        for dist in distributions:
            dist_df = temp_df[temp_df['f'] == dist]
            
            if not dist_df.empty:
                # Sort by n_target
                dist_df = dist_df.sort_values(by='$n_{target}$')
                
                plt.plot(
                    dist_df['$n_{target}$'],
                    dist_df['S'],
                    color=colors[dist],
                    linestyle=styles[dist],
                    linewidth=2,
                    marker='o',
                    markersize=6,
                    label=labels[dist]
                )
        
        plt.xlabel('Target Density $n$', fontsize=14)
        plt.ylabel('Entropy $S/k_B$', fontsize=14)
        plt.title(f'Entropy vs Density\\nT = {temperature:.4f}|t|, t = {t}, t\' = {tp}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"S_vs_n_T{temperature:.4f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Entropy vs density plot saved: {filename}")

def run_extended_simulation(
    temperatures: List[float],
    n_target_list: List[float],
    h_values: np.ndarray,
    distributions: List[str] = ("FD", "FLB", "NFL"),
    Nk: int = 100,
    t: float = -1.0,
    tp: float = 0.25,
    a: float = 1.0,
    out_root: str = "plots_extended_freeenergy",
    plot_all: bool = True,
    n_target_per_dist: dict = None,
):
    """
    Extended simulation with entropy calculations and comprehensive plotting.
    
    This version computes entropy and generates plots for:
    - Chemical potential vs temperature
    - Chemical potential vs density
    - Entropy vs temperature  
    - Entropy vs density
    - All existing plots (magnetization, distributions, etc.)
    """
    
    print(f"\n{'='*80}")
    print(f"EXTENDED FREEENERGY SIMULATION WITH ENTROPY")
    print(f"{'='*80}")
    print(f"Temperatures: {temperatures}")
    print(f"n_targets: {n_target_list if not n_target_per_dist else 'per-distribution'}")
    print(f"h_values: {h_values}")
    print(f"Distributions: {distributions}")
    print(f"Grid size: {Nk}x{Nk}")
    
    # Setup output directory
    timestamp = f"T{temperatures[0]:.5f}" if len(temperatures) == 1 else f"T{temperatures[0]:.3f}-{temperatures[-1]:.3f}"
    output_dir = f"{out_root}_{timestamp}_t{t}_tp{tp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Color scheme
    colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    styles = {"FD": "-", "FLB": "--", "NFL": "-."}
    labels = {"FD": "Fermi-Dirac", "FLB": "Fermi Liquid", "NFL": "Non-Fermi Liquid"}
    
    # Storage for results
    global_rows = []
    mag_vs_h = {dist: [] for dist in distributions}
    h_values_list = []  # Store h values for plotting
    
    # Determine which approach to use
    use_per_dist = n_target_per_dist is not None
    
    # Loop over parameters
    for h in h_values:
        print(f"\n  Magnetic field h = {h:.8f}")
        
        h_folder = os.path.join(output_dir, f"h_{h:.10f}")
        os.makedirs(h_folder, exist_ok=True)
        
        rows_h = []
        magnetizations_h = {dist: [] for dist in distributions}  # Store magnetizations for this h
        
        for T in temperatures:
            print(f"    Temperature T = {T:.6f}")
            
            if use_per_dist:
                # Per-distribution approach
                target_combinations = []
                for dist in distributions:
                    if dist in n_target_per_dist:
                        for n_val in n_target_per_dist[dist]:
                            target_combinations.append((dist, n_val))
            else:
                # Traditional approach - same n_target for all distributions
                target_combinations = []
                for n_target in n_target_list:
                    for dist in distributions:
                        target_combinations.append((dist, n_target))
            
            # Process each (distribution, n_target) combination
            for dist, n_target in target_combinations:
                print(f"      {dist}: n_target = {n_target:.6f}")
                
                # Create k-space grid
                kx = np.linspace(0, 2 * np.pi / a, Nk, endpoint=False)
                ky = kx.copy()
                
                # Solve for chemical potential
                result = one_state(dist, T, n_target, h, kx, ky, t, tp, a)
                mu = result['mu']
                n_up = result['n_up']
                n_down = result['n_down']
                
                # Compute entropy
                entropy = compute_total_entropy(T, h, mu, Nk, t, tp, a, dist, n_target)
                
                # Store magnetization for this h value
                magnetization = n_up - n_down
                magnetizations_h[dist].append(magnetization)
                
                # Store results
                row = {
                    'f': dist,
                    'T': T,
                    '$n_{target}$': n_target,
                    'h': h,
                    'mu': mu,
                    'n_up': n_up,
                    'n_down': n_down,
                    'S': entropy,  # Add entropy to results
                }
                rows_h.append(row)
                global_rows.append(row)
                
                print(f"        μ = {mu:.6f}, S = {entropy:.6f}, M = {magnetization:.6f}")
        
        # Calculate average magnetization for each distribution at this h value
        for dist in distributions:
            if magnetizations_h[dist]:
                M_avg = np.mean(magnetizations_h[dist])
                mag_vs_h[dist].append(M_avg)
                print(f"    {dist:<3}: ⟨M⟩ = {M_avg:.6f}")
            else:
                mag_vs_h[dist].append(0.0)
        
        h_values_list.append(h)
        
        # Create DataFrame for this h value
        df_h = pd.DataFrame(rows_h)
        
        # Generate plots for this h value
        if len(df_h) > 0:
            plot_spin_polarization(df_h, h_folder, distributions, t, tp)
    
    # Generate aggregated magnetization vs h plot
    print(f"\nGenerating magnetization vs h plot with {len(h_values_list)} h values")
    
    plt.figure(figsize=(12, 8))
    plotted_any = False
    
    for dist in distributions:
        if len(mag_vs_h[dist]) == len(h_values_list) and len(h_values_list) > 1:
            plt.plot(np.array(h_values_list) * 1e4, mag_vs_h[dist], styles[dist], marker="o", 
                    color=colors[dist], label=labels[dist], linewidth=3, markersize=8)
            plotted_any = True
            print(f"  Plotted {dist}: {len(mag_vs_h[dist])} points, M range: {min(mag_vs_h[dist]):.6f} to {max(mag_vs_h[dist]):.6f}")
    
    if plotted_any:
        plt.xlabel(r"Zeeman field, $h/|t| \times 10^{4}$", fontsize=16)
        plt.ylabel("Magnetic moment, $⟨n_↑ - n_↓⟩$", fontsize=16)
        title_suffix = "per-distribution targets" if use_per_dist else "same targets"
        plt.title(f"Average magnetization vs h ({title_suffix})", fontsize=18)
        
        # Add legend with professional styling
        legend = plt.legend(fontsize=14, loc='upper left', frameon=False)
        
        # Professional tick styling
        plt.tick_params(
            axis='both',
            which='both',
            direction='in',
            bottom=True, top=True,
            left=True, right=True,
            labelsize=15,
            width=3,
            length=8,
            pad=6
        )
        
        # Manual scaling - x-axis values should be scaled by 1e4 for the label to be correct
        # Since label says "× 10^4", the actual plotted values should be h*1e4  
        # No automatic scientific notation formatting since we handle it manually in the label
        
        # Set reasonable number of ticks
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=6))
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
        
        # Add temperature annotation (no background box for professional look)
        temp_value = temperatures[0] if temperatures else 0.001
        temp_sci = format_temperature_scientific(temp_value)
        plt.annotate(f'$\\frac{{k_B T}}{{|t|}} = {temp_sci}$', 
                    xy=(0.05, 0.75), xytext=(0.05, 0.85),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=16, ha='left', va='top')
        
        plt.grid(alpha=0.3)
        
        # Improve scaling for small values
        y_min = min([min(mag_vs_h[dist]) for dist in distributions]) 
        y_max = max([max(mag_vs_h[dist]) for dist in distributions])
        y_range = y_max - y_min
        if y_range > 0:
            plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # Use regular decimal notation for y-axis (values are 0 to 1)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Aggregated_M_vs_h.png"), dpi=300, bbox_inches='tight')
        print(f"Magnetization vs h plot saved: {os.path.join(output_dir, 'Aggregated_M_vs_h.png')}")
        
        # Also save a version with value annotations for small data
        plt.figure(figsize=(12, 8))
        for dist in distributions:
            if len(mag_vs_h[dist]) == len(h_values_list):
                plt.plot(np.array(h_values_list) * 1e4, mag_vs_h[dist], styles[dist], marker="o", 
                        color=colors[dist], label=labels[dist], linewidth=3, markersize=8)
                
                # Add value annotations for non-zero points
                h_array = np.array(h_values_list) * 1e4
                #for i, (h, M) in enumerate(zip(h_array, mag_vs_h[dist])):
                #    if abs(M) > 1e-6:  # Only annotate non-zero values
                #        plt.annotate(f'{M:.4f}', (h * 1e4, M), xytext=(5, 5), 
                #                   textcoords='offset points', fontsize=9, 
                #                   color=colors[dist], alpha=0.8)
        
        plt.xlabel(r"Zeeman field, $h/|t| \times 10^{4}$", fontsize=16)
        plt.ylabel("Magnetic moment, $⟨n_↑ - n_↓⟩$", fontsize=16)
        #plt.title(f"Magnetization vs h with Values ({title_suffix})", fontsize=18)
        
        # Add legend with professional styling
        legend = plt.legend(fontsize=14, loc='upper left', frameon=False)
        
        # Add temperature label in scientific notation using annotate
        temp_value = temperatures[0] if temperatures else 0.001
        temp_sci = format_temperature_scientific(temp_value)
        plt.annotate(f'$\\frac{{k_B T}}{{|t|}} = {temp_sci}$', 
                    xy=(0.05, 0.75), xytext=(0.05, 0.85),
                    xycoords='axes fraction', textcoords='axes fraction',
                    fontsize=12, ha='left', va='top')
        
        plt.grid(alpha=0.3)
        
        # Same scaling as above
        if y_range > 0:
            plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        # Use regular decimal notation for y-axis (values are 0 to 1)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Aggregated_M_vs_h_ANNOTATED.png"), dpi=300, bbox_inches='tight')
        print(f"Annotated magnetization plot saved: {os.path.join(output_dir, 'Aggregated_M_vs_h_ANNOTATED.png')}")
        
    else:
        print("No magnetization data to plot - check if multiple h values are provided")
    
    plt.close()
    
    # Create comprehensive DataFrame
    df_all = pd.DataFrame(global_rows)
    df_all.to_csv(os.path.join(output_dir, "results_raw.csv"), index=False)
    
    # Generate comprehensive plots if requested
    if plot_all and len(df_all) > 0:
        print(f"\n{'='*60}")
        print(f"GENERATING COMPREHENSIVE PLOTS")
        print(f"{'='*60}")
        
        # Chemical potential plots
        plot_chemical_potential_vs_temperature(df_all, output_dir, distributions, t, tp)
        plot_chemical_potential_vs_density(df_all, output_dir, distributions, t, tp)
        
        # Entropy plots
        plot_entropy_vs_temperature(df_all, output_dir, distributions, t, tp)
        plot_entropy_vs_density(df_all, output_dir, distributions, t, tp)
    
    print(f"\nExtended simulation completed!")
    print(f"Results saved in: {output_dir}")
    print(f"Raw data: {os.path.join(output_dir, 'results_raw.csv')}")

if __name__ == "__main__":
    # Uncomment the line below to run the test
    # test_n_target_initial_guess()
    
    ###########################################################################
    #                         SIMULATION CONFIGURATION                        #
    ###########################################################################
    
    # Basic parameters
    temperatures = [0.01]
    h_values = np.array([0.0])
    distributions = ["FD", "FLB", "NFL"]
    Nk = 100  # Grid size (use larger for production runs)
    
    # =========================================================================
    # COMPARISON MODE SELECTION - Choose one of the following modes:
    # =========================================================================
    
    mode = "extended_entropy"  # Options: "same_density", "same_filling", "test_plots", "extended_entropy"
    
    if mode == "same_density":
        # MODE 1: Traditional approach - same n_target for all distributions
        print("Running SAME DENSITY comparison")
        n_target_list = [0.8, 1.2]  # Same density for all distributions
        
        run_simulation(
            temperatures=temperatures,
            n_target_list=n_target_list,
            h_values=h_values,
            distributions=distributions,
            Nk=Nk,
            plot_distributions=True
        )
        
    elif mode == "same_filling":
        # MODE 2: Same band filling - physically meaningful comparison
        print("Running SAME BAND FILLING comparison")
        
        # Define band filling fractions you want to compare
        band_fillings = [0.4, 0.8]  # 40% and 80% of maximum capacity
        
        # Calculate appropriate n_targets for each distribution
        max_occupancies = {"FD": 2.0, "FLB": 2.0, "NFL": 1.0}
        
        # Create n_target dictionary for each distribution
        n_target_per_dist = {}
        for dist in distributions:
            n_target_per_dist[dist] = [bf * max_occupancies[dist] for bf in band_fillings]
        
        print("Band filling to n_target conversion:")
        for bf in band_fillings:
            print(f"  {bf:.1%} filling:")
            for dist in distributions:
                n_val = bf * max_occupancies[dist]
                print(f"    {dist}: n_target = {n_val:.3f} (max_occ = {max_occupancies[dist]})")
        
        run_simulation(
            temperatures=temperatures,
            n_target_list=[],  # Not used when n_target_per_dist is provided
            h_values=h_values,
            distributions=distributions,
            Nk=Nk,
            plot_distributions=True,
            n_target_per_dist=n_target_per_dist
        )
        
    elif mode == "test_plots":
        # MODE 3: Test plotting functions only
        print("Running test plots")
        test_distribution_plots()
        
    elif mode == "extended_entropy":
        # MODE 4: Extended simulation with entropy calculations and comprehensive plotting
        print("Running EXTENDED SIMULATION with entropy and comprehensive plots")
        
        # Extended parameter setup - restore multiple T and n for comprehensive plots
        temperatures_extended = [0.001, 0.01, 0.05]  # Multiple temperatures for T-dependence
        
        # Multiple band fillings for density dependence plots
        band_fillings = [0.3, 0.5, 0.7]  # 30%, 50%, 70% of maximum capacity
        max_occupancies = {"FD": 2.0, "FLB": 2.0, "NFL": 1.0}
        
        # Create n_target dictionary for each distribution
        n_target_per_dist = {}
        for dist in distributions:
            n_target_per_dist[dist] = [bf * max_occupancies[dist] for bf in band_fillings]
        
        print("Band filling to n_target conversion for extended simulation:")
        for bf in band_fillings:
            print(f"  {bf:.1%} filling:")
            for dist in distributions:
                n_val = bf * max_occupancies[dist]
                print(f"    {dist}: n_target = {n_val:.3f}")
        
        # Multiple h values for magnetization curves - smaller range for faster computation
        h_values_extended = np.linspace(0.0, 0.0005, 6)  # 6 points from 0 to 0.0005
        print(f"\nMagnetic field values: {h_values_extended}")
        
        # Reduced lattice size for manageable computation with expanded parameter space
        Nk_small = 8  # Even smaller grid for comprehensive parameter sweep
        print(f"Using {Nk_small}x{Nk_small} lattice for comprehensive parameter sweep")
        
        # Run extended simulation
        run_extended_simulation(
            temperatures=temperatures_extended,
            n_target_list=[],  # Not used when n_target_per_dist is provided
            h_values=h_values_extended,
            distributions=distributions,
            Nk=Nk_small,  # Use smaller grid
            plot_all=True,  # Generate all comprehensive plots
            n_target_per_dist=n_target_per_dist
        )

# Add this helper function near the top of the file after imports
def format_temperature_scientific(temp_value):
    """Convert temperature to proper mathematical scientific notation (10^{-n}) instead of CS notation (1e-n)"""
    import math
    if temp_value == 0:
        return "0"
    
    # Get the exponent
    exponent = int(math.floor(math.log10(abs(temp_value))))
    
    # Get the mantissa
    mantissa = temp_value / (10 ** exponent)
    
    # For clean notation, if mantissa is 1.0, just show 10^{exponent}
    if abs(mantissa - 1.0) < 1e-10:
        return f"10^{{{exponent}}}"
    else:
        # Otherwise show mantissa × 10^{exponent}
        return f"{mantissa:.1f} \\times 10^{{{exponent}}}"

def format_magnetic_field(h_val):
    """Format magnetic field value for display with LaTeX scientific notation."""
    if h_val == 0:
        return "0"
    elif h_val < 0.001:
        # Use scientific notation for very small values
        exponent = int(np.floor(np.log10(abs(h_val))))
        mantissa = h_val / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            return rf"10^{{{exponent}}}"
        else:
            return rf"{mantissa:.1f} \cdot 10^{{{exponent}}}"
    elif h_val < 0.01:
        # Use scientific notation for small values
        exponent = int(np.floor(np.log10(abs(h_val))))
        mantissa = h_val / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            return rf"10^{{{exponent}}}"
        else:
            return rf"{mantissa:.1f} \cdot 10^{{{exponent}}}"
    elif h_val < 0.1:
        # Use scientific notation for medium values
        exponent = int(np.floor(np.log10(abs(h_val))))
        mantissa = h_val / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            return rf"10^{{{exponent}}}"
        else:
            return rf"{mantissa:.1f} \cdot 10^{{{exponent}}}"
    else:
        # Use decimal notation for larger values
        formatted = f"{h_val:.3f}".rstrip('0').rstrip('.')
        return formatted

def add_plot_annotations(ax, T, h, n_target, x_pos=0.02, y_pos=0.02):
    """Add parameter annotations to plots instead of titles."""
    # Format temperature with scientific notation
    if T < 0.001:
        # Use scientific notation for very small temperatures
        exponent = int(np.floor(np.log10(abs(T))))
        mantissa = T / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            temp_text = rf"$\frac{{k_B T}}{{|t|}} = 10^{{{exponent}}}$"
        else:
            temp_text = rf"$\frac{{k_B T}}{{|t|}} = {mantissa:.1f} \cdot 10^{{{exponent}}}$"
    elif T < 0.01:
        # Use scientific notation for small temperatures
        exponent = int(np.floor(np.log10(abs(T))))
        mantissa = T / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            temp_text = rf"$\frac{{k_B T}}{{|t|}} = 10^{{{exponent}}}$"
        else:
            temp_text = rf"$\frac{{k_B T}}{{|t|}} = {mantissa:.1f} \cdot 10^{{{exponent}}}$"
    elif T < 0.1:
        # Use scientific notation for medium temperatures
        exponent = int(np.floor(np.log10(abs(T))))
        mantissa = T / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            temp_text = rf"$\frac{{k_B T}}{{|t|}} = 10^{{{exponent}}}$"
        else:
            temp_text = rf"$\frac{{k_B T}}{{|t|}} = {mantissa:.1f} \cdot 10^{{{exponent}}}$"
    else:
        # Use decimal notation for larger temperatures
        formatted = f"{T:.3f}".rstrip('0').rstrip('.')
        temp_text = rf"$\frac{{k_B T}}{{|t|}} = {formatted}$"
    
    # Format magnetic field with proper LaTeX fraction notation
    h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h)}$"
    
    # Format band filling
    n_text = f"$n = {n_target:.2f}$"
    
    # Combine annotations
    annotation_text = f"{temp_text}\n{h_text}\n{n_text}"
    
    ax.text(x_pos, y_pos, annotation_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom', horizontalalignment='left')

# Global worker functions for parallelization
def worker_FD(args):
    """Global worker function for FD calculations."""
    kx, ky, T, h, mu, t, tp, a, n_target = args
    sol = nk_FD(kx, ky, T, h, mu, t, tp, a, n_target)
    return sol[0], sol[1]

def worker_FLB(args):
    """Global worker function for FLB calculations."""
    kx, ky, T, h, mu, t, tp, a, n_target = args
    return nk_FLB(kx, ky, T, h, mu, t, tp, a, n_target)

def worker_NFL(args):
    """Global worker function for NFL calculations."""
    kx, ky, T, h, mu, t, tp, a, n_target = args
    return nk_NFL(kx, ky, T, h, mu, t, tp, a, n_target)

def worker_entropy(args):
    """Global worker function for entropy calculations."""
    kx, ky, T, h, mu, t, tp, a, distribution, n_target = args
    
    # Get occupations at this k-point
    if distribution == "FD":
        sol = nk_FD(kx, ky, T, h, mu, t, tp, a, n_target)
    elif distribution == "FLB":
        sol = nk_FLB(kx, ky, T, h, mu, t, tp, a, n_target)
    elif distribution == "NFL":
        sol = nk_NFL(kx, ky, T, h, mu, t, tp, a, n_target)
    else:
        sol = [0.0, 0.0]
    
    n_up, n_down = sol[0], sol[1]
    
    # Return entropy contribution from this k-point
    return entropy_k_point(n_up, n_down, T, distribution)

# Test function to verify n_target/2 initial guess implementation
def test_n_target_initial_guess():
    """
    Test function to verify that the n_target/2 initial guess is working correctly.
    """
    print("Testing n_target/2 initial guess implementation...")
    
    # Test parameters
    T = 0.01
    h = 0.0
    mu = 0.0
    t = -1.0
    tp = 0.25
    a = 1.0
    kx = 0.5
    ky = 0.5
    
    # Test different n_target values
    n_target_values = [0.2, 0.5, 0.8, 1.0, 1.5]
    
    for n_target in n_target_values:
        print(f"\nTesting n_target = {n_target}")
        
        # Test FD
        try:
            result_fd = nk_FD(kx, ky, T, h, mu, t, tp, a, n_target)
            print(f"  FD: n_up = {result_fd[0]:.6f}, n_down = {result_fd[1]:.6f}, total = {result_fd[0] + result_fd[1]:.6f}")
        except Exception as e:
            print(f"  FD: Error - {e}")
        
        # Test FLB
        try:
            result_flb = nk_FLB(kx, ky, T, h, mu, t, tp, a, n_target)
            print(f"  FLB: n_up = {result_flb[0]:.6f}, n_down = {result_flb[1]:.6f}, total = {result_flb[0] + result_flb[1]:.6f}")
        except Exception as e:
            print(f"  FLB: Error - {e}")
        
        # Test NFL
        try:
            result_nfl = nk_NFL(kx, ky, T, h, mu, t, tp, a, n_target)
            print(f"  NFL: n_up = {result_nfl[0]:.6f}, n_down = {result_nfl[1]:.6f}, total = {result_nfl[0] + result_nfl[1]:.6f}")
        except Exception as e:
            print(f"  NFL: Error - {e}")
    
    print("\nTest completed!")
"""
if __name__ == "__main__":
    # Uncomment the line below to run the test
    # test_n_target_initial_guess()
    
    ###########################################################################
    #                         SIMULATION CONFIGURATION                        #
    ###########################################################################
"""
