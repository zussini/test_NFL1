#!/usr/bin/env python3
"""
Test for NFL metamagnetic transition at ultra-low temperature T = 0.00001.
Enhanced with comprehensive thermodynamic plots and BOTH self-consistent 
and comparison distribution plots.
"""

import sys
import os
sys.path.append('.')

from chartformat_freeenergy_SPIN_ALMOST_Correct_Magnetization_Entropy3 import (
    run_extended_simulation, 
    one_state,
    compute_total_entropy,
    plot_chemical_potential_vs_temperature,
    plot_chemical_potential_vs_density,
    plot_entropy_vs_temperature,
    plot_entropy_vs_density,
    plot_all_distributions,
    plot_distributions_per_dist,
    plot_distribution_vs_energy,
    plot_distribution_vs_momentum,
    plot_distribution_heatmap,
    compute_distribution_data,
    solve_mu_FD,  # This is the key function for comparison plots
    solve_mu_FLB,
    solve_mu_NFL,
    nk_FD,
    nk_FLB,
    nk_NFL,
    tight_binding_dispersion
)
from spin_split_wrappers import (
    one_state_split,
    compute_total_entropy_split,
    one_state_split_self_consistent,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Parallelization configuration
USE_PARALLEL = True  # Set to False to use sequential processing
N_CORES = None  # Set to None for all cores, or specify a number (e.g., 4)

# Configuration section for easy customization
CONFIG = {
    'parallel': {
        'enabled': USE_PARALLEL,
        'n_cores': N_CORES,
        'chunk_size': 1,  # Number of tasks per worker (can be optimized)
    },
    'self_consistency': {
        'enabled': True,
        'max_iter': 50,
        'tol': 1e-6,
        'damping': 0.5,
        'q_bounds': (0.05, 5.0),
        'fallback_on_failure': True,
    },
    'testing': {
        'run_test': False,  # Set to True to run parallel implementation test
        'test_parameters': {
            'temperatures': [0.001],
            'h_values': [0.0, 0.0007],
            'n_targets': [0.50, 0.95],
            #'n_targets': [0.96, 0.98],
            'distributions': ["FD", "FLB", "NFL"],
            'Nk': 100
        }
    },
    'performance': {
        'progress_update_interval': 10,  # Show progress every N calculations
        'detailed_output': True,  # Print detailed calculation info
    }
}

# Update global variables from config
USE_PARALLEL = CONFIG['parallel']['enabled']
N_CORES = CONFIG['parallel']['n_cores']

# Set global matplotlib parameters for better plots
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 24,
    'figure.titlesize': 28,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'xtick.top': True,
    'ytick.right': True,
})

print("Testing NFL METAMAGNETIC TRANSITION at ULTRA-LOW T = 0.00001...")
print("Approaching quantum limit with enhanced thermodynamic analysis")

# Parameters for ultra-low temperature test
temperatures_test = [0.0001]  # Ultra-low temperature
distributions = ["FD", "FLB", "NFL"]
temperatures_test = [0.001]
# Focus on critical region around h ≈ 0.00075
h_values_critical = np.linspace(0.00068, 0.00078, 50)  # Same range for comparison
h_values_critical = np.linspace(0.00070720, 0.00070735, 50)  # Same range for comparison
h_values_critical = np.linspace(0.00070884, 0.00070888, 9)  # Same range for comparison
#h_values_critical = np.linspace(0.00070715, 0.00070735, 90)  # Same range for compariso
print(f"H values: {h_values_critical}")
print(f"Expected NFL quantum transition around h ≈ 0.00075")
Nk_path = 100  # Number of points along the path
Nk_test = 100 # Small grid for speed but enough for distributions
Nk = 100
N_L = 100
Nk_inset = 100
inset_samples = 5
# Test parameters
T = 0.01
h = 0.0
n_target = 0.8
t = -1.0
tp = 0.25
a = 1.0
# Explicit spin-split hoppings for strict control (no fallbacks)
t_up = t
tp_up = tp
t_dn = 0.8*t_up
tp_dn = 0.8*tp_up
# Multiple band fillings for comprehensive analysis
band_fillings = [0.50, 0.95]  # Multiple band filling values
#band_fillings = [0.96, 0.98]  # Multiple band filling values
#band_fillings = [0.95]
max_occupancies = {"FD": 1.0, "FLB": 1.0, "NFL": 1.0}  # All should be 1.0 for consistency

n_target_per_dist = {}
for dist in distributions:
    n_target_per_dist[dist] = [bf * max_occupancies[dist] for bf in band_fillings]

print(f"Band fillings: {band_fillings}")
print(f"n_target_per_dist: {n_target_per_dist}")

# Temporarily disable the long simulation for testing

# --- INACTIVE SIMULATION ENTRYPOINT: run_extended_simulation(...) is commented out ---
#run_extended_simulation(
#    temperatures=temperatures_test,
#    n_target_list=[],
#    h_values=h_values_critical,
#    distributions=distributions,
#    Nk=Nk_test,
#    out_root="quantum_limit_NFL_test",
#    plot_all=False,  # Skip other plots for speed
#    n_target_per_dist=n_target_per_dist
#)


print("\n" + "="*60)
print("GENERATING COMPREHENSIVE THERMODYNAMIC ANALYSIS")
print("="*60)

# Enhanced parameters for extended analysis with temperature-specific fields
temperatures_extended = [0.0,0.0001]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points
temperatures_extended = [0.01,0.05,0.1,0.3,0.5]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points
temperatures_extended = [0.01,0.02,0.03]#,0.04,0.05]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points
temperatures_extended = [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.055,0.06,0.07,0.08,0.09,0.10,0.11]#,0.04,0.05]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points
temperatures_extended = [0.0001,0.001,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11]#,0.01,0.02]#,0.04,0.05]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points
#temperatures_extended = [0.0001,0.001]
#temperatures_extended = [0.1,0.11]
#temperatures_extended = [0.0001,0.01]

#temperatures_extended = [0.07,0.08,0.09,0.10,0.11]#,0.04,0.05]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points
#temperatures_extended = [0.05,0.051,0.052,0.053,0.054,0.055,0.056,0.057,0.058,0.059,0.06]#,0.04,0.05]#, 0.001]#, 0.01, 0.1, 1.0, 8.0]  # Key temperature points

#n_targets_extended = [0.95]
n_targets_extended = [0.50, 0.95]
#n_targets_extended = [0.96, 0.98]
#n_targets_extended = [0.95]
# Temperature-specific magnetic field ranges for NFL metamagnetic transitions
TEMPERATURE_SPECIFIC_FIELDS ={
    0.0001: np.linspace(0.00070720, 0.00070735, 5),
    1e-3: np.linspace(0.00070884, 0.00070888, 5),
    0.01: np.linspace(0.00070, 0.00100, 5),  # Very low T: wider range for exploration
    0.1: np.linspace(0.0000, 0.025, 5),  # Low T: broader range
    1.0: np.linspace(0.0000, 0.20, 5),  # Medium T: even broader range
    8.0: np.linspace(0.0000, 1.0000, 5),  # High T: very broad range
}
TEMPERATURE_SPECIFIC_FIELDS = {
    0.0 : np.linspace(0.0,0.0008,5),
    0.0001: np.linspace(0.000700438, 0.00070045, 5),
    #1e-3: np.linspace(0.000701995, 0.000702005, 50),
    #0.01: np.linspace(0.00070, 0.00100, 5),  # Very low T: wider range for exploration
    #0.1: np.linspace(0.0000, 0.025, 5),  # Low T: broader range
    #1.0: np.linspace(0.0000, 0.20, 5),  # Medium T: even broader range
    #8.0: np.linspace(0.0000, 1.0000, 5),
}
TEMPERATURE_SPECIFIC_FIELDS = {
    #0.0 : np.linspace(0.0,0.0008,5),
    #0.0001: np.linspace(0.000700438, 0.00070045, 5),
    #1e-3: np.linspace(0.000701995, 0.000702005, 50),
    0.01: np.linspace(0.00000, 0.00100, 5),  # Very low T: wider range for exploration
    0.05: np.linspace(0.00000, 0.00100, 5),  # Very low T: wider range for exploration
    0.1: np.linspace(0.0000, 0.001, 5),  # Low T: broader range
    0.3: np.linspace(0.0000, 0.001, 5),  # Medium T: even broader range
    0.5: np.linspace(0.0000, 0.0010, 5),
}
TEMPERATURE_SPECIFIC_FIELDS = {
    #0.0 : np.linspace(0.0,0.0008,5),
    #0.0001: np.linspace(0.000700438, 0.00070045, 5),
    #1e-3: np.linspace(0.000701995, 0.000702005, 50),
    0.01: np.linspace(0.0007178, 0.000718, 100),  # Very low T: wider range for exploration
    0.02: np.linspace(0.00073625, 0.0007363, 100),  # Very low T: wider range for exploration
    0.03: np.linspace(0.00075545, 0.0007555, 100),  # Low T: broader range
    #0.04: np.linspace(0.0004, 0.0008, 5),  # Medium T: even broader range
    #0.05: np.linspace(0.0004, 0.0008, 5),
}
TEMPERATURE_SPECIFIC_FIELDS = {
    #0.0 : np.linspace(0.0,0.0008,5),
    0.0001: np.linspace(0.000700438, 0.00070045, 50),
    1e-3: np.linspace(0.000701995, 0.000702005, 50),
    #0.01: np.linspace(0.000717884, 0.000717888, 50),  # Very low T: wider range for exploration
    #0.02: np.linspace(0.0007362717, 0.000736274, 50),  # Very low T: wider range for exploration
    #0.03: np.linspace(0.000755479, 0.000755481, 50),  # Low T: broader range
    #0.04: np.linspace(0.000775560, 0.000775565, 50),  # Medium T: even broader range
    #0.05: np.linspace(0.000796577, 0.00079658, 50),
    #0.055: np.linspace(0.000807487347, 0.000814487347, 50),
    #0.06: np.linspace(0.000817487347, 0.0008285, 50),
    #0.07: np.linspace(0.000841668, 0.0008416710,50),
    #0.08: np.linspace(0.000865880, 0.000865905, 50),
    #0.09: np.linspace(0.000891335, 0.000891336, 50),
    #0.10: np.linspace(0.0009180945, 0.0009181005, 50),
    #0.11: np.linspace(0.0009462752, 0.000946277, 50),
}
TEMPERATURE_SPECIFIC_FIELDS = {
    #0.0 : np.linspace(0.0,0.0008,5)0.00072544,
    0.0001: 2*np.linspace(0.00070685, 0.0007074, 30),
    1e-3: 2*np.linspace(0.00070858, 0.000708950, 30),
    #0.01: 2*np.linspace(0.000717884+0.000005, 0.000728888+0.000006, 30),  # Very low T: wider range for exploration
    #0.02: 2*np.linspace(0.0007362717+0.000005, 0.000747274+0.000006, 30),  # Very low T: wider range for exploration
    #0.03: 2*np.linspace(0.000755479+0.000003, 0.000755481+0.000010, 30),  # Low T: broader range
    #0.04: 2*np.linspace(0.000775560+0.000003, 0.000775565+0.000010, 30),  # Medium T: even broader range
    #0.05: 2*np.linspace(0.000796577+0.000003, 0.00079658+0.000011, 30),
    #0.055: 2*np.linspace(0.000807487347+0.000005, 0.000814487347+0.000007, 5),
    #0.06: 2*np.linspace(0.000817487347+0.000005, 0.0008285+0.000010, 30),
    #0.07: 2*np.linspace(0.000841668+0.000003, 0.0008416710+0.000010,30),
    #0.08: 2*np.linspace(0.000865880+0.000003, 0.000865905+0.000011, 30),
    #0.09: 2*np.linspace(0.000891335+0.000003, 0.000891336+0.000012, 30),
    #0.10: 2*np.linspace(0.0009180945+0.000003, 0.0009181005+0.000012, 30),
    #0.11: 2*np.linspace(0.0009462752+0.000003, 0.000946277+0.000012, 30),
}
#finding with h 5 values the exact ranges for the transition  cirrected n_target initial
TEMPERATURE_SPECIFIC_FIELDS = {
    #0.0 : np.linspace(0.0,0.0008,5)0.00072544,
    0.0001: np.linspace(0.0014139, 0.0014148, 50),
    1e-3: np.linspace(0.00141716,0.00141720, 50),
    #0.01: np.linspace(0.001448,0.001450000 , 50),  # Very low T: wider range for exploration
    #0.02: np.linspace(0.001486 ,0.001489 , 50),  # Very low T: wider range for exploration
    #0.03: np.linspace(0.001526 , 0.001527 , 50),  # Low T: broader range
    #0.04: np.linspace(0.001565 ,0.00157 , 50),  # Medium T: even broader range
    #0.05: np.linspace( 0.001607,0.001615 , 50),
    #0.055: 2*np.linspace(0.000807487347+0.000005, 0.000814487347+0.000007, 5),
    #0.06: np.linspace(0.0016470,0.001658 , 50),
    #0.07: np.linspace( 0.001675336 ,0.001723342  ,50),
    #0.08: np.linspace(0.001725 ,0.00175381  , 50),
    #0.09: np.linspace(0.00178267  ,  0.001816 , 50),
    #0.10: np.linspace(0.001842189 , 0.001850202 , 50),
    #0.11: np.linspace(0.0019100000, 0.001922000  , 50),
}
# Per-(n,T) overrides. Using the same ranges for n=0.50 and n=0.95 as requested.
TEMPERATURE_SPECIFIC_FIELDS_BY_N = {
    0.50: {
        0.0001: np.linspace(0.001414552, 0.00141457, 5),
        0.0010: np.linspace(0.00141774, 0.00141778, 5),
        0.01: np.linspace(0.001448, 0.001450000, 5),
        0.02: np.linspace(0.001486, 0.001489, 5),
        0.03: np.linspace(0.001526, 0.001527, 5),
        0.04: np.linspace(0.001565, 0.001570, 5),
        0.05: np.linspace(0.001607, 0.001615, 5),
        0.06: np.linspace(0.0016470, 0.001658, 5),
        0.07: np.linspace(0.001675336, 0.001723342, 5),
        0.08: np.linspace(0.001725, 0.00175381, 5),
        0.09: np.linspace(0.00178267, 0.001816, 5),
        0.10: np.linspace(0.001842189, 0.001850202, 5),
        0.11: np.linspace(0.0019100000, 0.001922000, 5),
    },
    0.95: {
        0.0001: np.linspace(0.00141394, 0.00141408, 5),
        0.0010: np.linspace(0.001417184, 0.001417194, 5),
        0.01: np.linspace(0.001448, 0.001450000, 5),
        0.02: np.linspace(0.001486, 0.001489, 5),
        0.03: np.linspace(0.001526, 0.001527, 5),
        0.04: np.linspace(0.001565, 0.001570, 5),
        0.05: np.linspace(0.001607, 0.001615, 5),
        0.06: np.linspace(0.0016470, 0.001658, 5),
        0.07: np.linspace(0.001675336, 0.001723342, 5),
        0.08: np.linspace(0.001725, 0.00175381, 5),
        0.09: np.linspace(0.00178267, 0.001816, 5),
        0.10: np.linspace(0.001842189, 0.001850202, 5),
        0.11: np.linspace(0.0019100000, 0.001922000, 5),
    },
}
# Per-(n,T) overrides. Using the same ranges for n=0.50 and n=0.95 as requested.
"""TEMPERATURE_SPECIFIC_FIELDS_BY_N = {
    0.50: {
        0.0001: np.linspace(0.0014145615, 0.001414564, 50),
        0.0010: np.linspace(0.001417750, 0.001417753, 50),
        0.01: np.linspace(0.00145032, 0.00145035, 50),
        0.02: np.linspace(0.001488091, 0.001488094, 50),
        0.03: np.linspace(0.00152756, 0.00152757, 50),
        0.04: np.linspace(0.00156888, 0.001568886, 50),
        #0.05: np.linspace(0.0016075, 0.0016125, 50),
        0.05: np.linspace(0.0016119, 0.0016124, 50),
        0.06: np.linspace(0.0016475, 0.001658, 50),
        #0.07: np.linspace(0.001640, 0.001720, 50),
        0.07: np.linspace(0.001640, 0.001710, 50),
        0.08: np.linspace(0.001740, 0.001756, 50),
        0.09: np.linspace(0.001800, 0.0018082, 50),
        0.10: np.linspace(0.001845, 0.0018645, 50),
        0.11: np.linspace(0.001916, 0.0019225, 50),
    },
    0.95: {
        0.0001: np.linspace(0.00141395, 0.00141407, 50),
        0.0010: np.linspace(0.001417187, 0.0014171895, 50),
        0.01: np.linspace(0.00144973, 0.0014495, 50),
        0.02: np.linspace(0.001487443, 0.00148745, 50),
        0.03: np.linspace(0.001526878, 0.001526884, 50),
        0.04: np.linspace(0.001568144, 0.00156815, 50),
        0.05: np.linspace(0.001611373, 0.001611385, 50),
        0.06: np.linspace(0.0016567, 0.00165672, 50),
        0.07: np.linspace(0.00170425, 0.0017033, 50),
        0.08: np.linspace(0.00175424, 0.0017543, 50),
        0.09: np.linspace(0.001806855, 0.0018067, 50),
        0.10: np.linspace(0.0018622, 0.00186226, 50),
        0.11: np.linspace(0.001920633, 0.001920639, 50),
    },
}
# Per-(n,T) overrides. Using the same ranges for n=0.50 and n=0.95 as requested.
TEMPERATURE_SPECIFIC_FIELDS_BY_N = {
    0.50: {
        0.0001: np.linspace(0.0014145618, 0.0014145634, 50),
        0.0010: np.linspace(0.0014177507, 0.0014177516, 50),
        0.01: np.linspace(0.0014503389, 0.00145033953 50),
        0.02: np.linspace(0.001488089, 0.0014880900, 50),
        0.03: np.linspace(0.0015275638, 0.0015275641, 50),
        0.04: np.linspace(0.0015688791, 0.0015688799, 50),
        #0.05: np.linspace(0.0016075, 0.0016125, 50),
        0.05: np.linspace(0.00161215, 0.001612165, 50),
        0.06: np.linspace(0.0016475, 0.001658, 50),
        #0.07: np.linspace(0.001640, 0.001720, 50),
        0.07: np.linspace(0.001640, 0.001710, 50),
        0.08: np.linspace(0.001740, 0.001756, 50),
        0.09: np.linspace(0.001800, 0.0018082, 50),
        0.10: np.linspace(0.001845, 0.0018645, 50),
        0.11: np.linspace(0.001916, 0.0019225, 50),
    },
    0.95: {
        0.0001: np.linspace(0.00141395, 0.00141407, 50),
        0.0010: np.linspace(0.001417187, 0.0014171895, 50),
        0.01: np.linspace(0.001449735, 0.001449745, 50),
        0.02: np.linspace(0.001487448, 0.001487454, 50),
        0.03: np.linspace(0.0015268786, 0.0015268832, 50),
        0.04: np.linspace(0.001568147, 0.0015681492, 50),
        0.05: np.linspace(0.0016113755, 0.001611379, 50),
        0.06: np.linspace(0.0016567045, 0.001656708, 50),
        0.07: np.linspace(0.00170427, 0.00170430, 50),
        0.08: np.linspace(0.001754273, 0.00175428, 50),
        0.09: np.linspace(0.001806860, 0.001806868, 50),
        0.10: np.linspace(0.0018622241, 0.001862248, 50),
        0.11: np.linspace(0.0019206360, 0.0019206373, 50),
    },
}
"""
# Per-(n,T) overrides. Using the same ranges for n=0.50 and n=0.95 as requested.
TEMPERATURE_SPECIFIC_FIELDS_BY_N = {
    0.50: {
        #0.0001: np.linspace(0.0014145619, 0.0014145633, 2),
        0.0001: np.linspace(0.0014145619, 0.0014145633, 5),#,endpoint=True),
        0.0010: np.linspace(0.0014177508, 0.0014177515, 5),
        #0.01: np.linspace(0.00145033905, 0.00145033920, 2),
        0.01: np.linspace(0.00145033905, 0.00145033920, 5),#,endpoint=True),
        0.02: np.linspace(0.00148808875, 0.00148808905, 5),
        0.03: np.linspace(0.001527563805, 0.001527563835, 5),
        0.04: np.linspace(0.00156887945, 0.00156887947, 5),
        ##0.05: np.linspace(0.0016075, 0.0016125, 50),
        0.05: np.linspace(0.00161216175, 0.00161216195, 5),
        0.06: np.linspace(0.0016475, 0.001658, 5),
        ##0.07: np.linspace(0.001640, 0.001720, 50),
        0.07: np.linspace(0.001640, 0.001710, 5),
        0.08: np.linspace(0.001740, 0.001756, 5),
        0.09: np.linspace(0.001800, 0.0018082, 5),
        0.10: np.linspace(0.001845, 0.0018645, 5),
        0.11: np.linspace(0.001916, 0.0019225, 5),
    },
    0.95: {
        #0.0001: np.linspace(0.00141395, 0.00141407, 2),
        0.0001: np.linspace(0.00141395, 0.00141407, 5),#,endpoint=True),
        #0.0001: np.linspace(0.00141395, 0.001414008775510, 2, endpoint=True),
        #0.0001: np.linspace(0.001414001428571,0.001414028367347,6,endpoint=True),
        #0.0001: np.linspace(0.001414005,0.0014140080,2,endpoint=True),#MID H
        0.0010: np.linspace(0.001417187, 0.0014171895, 5),
        #0.01: np.linspace(0.0014497408, 0.0014497432, 2),
        0.01: np.linspace(0.0014497408, 0.0014497432, 5),#,endpoint=True),
        0.02: np.linspace(0.0014874498, 0.0014874515, 5),
        0.03: np.linspace(0.0015268795, 0.0015268823, 5),
        0.04: np.linspace(0.001568147, 0.0015681492, 5),
        0.05: np.linspace(0.0016113760, 0.0016113780, 5),
        0.06: np.linspace(0.0016567053, 0.001656707, 5),
        0.07: np.linspace(0.0017042837, 0.0017042853, 5),
        0.08: np.linspace(0.0017542762, 0.0017542775, 5),
        0.09: np.linspace(0.001806863, 0.0018068645, 5),
        0.10: np.linspace(0.0018622430, 0.001862245, 5),
        0.11: np.linspace(0.0019206361, 0.0019206371, 5),
    },    
    0.96: {
        0.0001: np.linspace(0.00141361, 0.00141376, 50 ),
        #0.0010: np.linspace(0.001416864, 0.001416878, 50),
        #0.0001: np.linspace(0.00141359, 0.00141380, 50 ),
        #0.0010: np.linspace(0.001416750, 0.001417, 50),        
        0.01: np.linspace(0.0014494508, 0.0014496932, 50),
        #0.02: np.linspace(0.0014871498, 0.0014874015, 50),
        #0.03: np.linspace(0.0015266095, 0.0015267823, 50),
        #0.04: np.linspace(0.001566047, 0.0015677992, 50),
        #0.05: np.linspace(0.0016110860, 0.0016113380, 50),
        #0.06: np.linspace(0.0016564153, 0.0016566570, 50),
        #0.07: np.linspace(0.0017039937, 0.0017042353, 50),
        #0.08: np.linspace(0.0017539862, 0.0017542275, 50),
        #0.09: np.linspace(0.001803963, 0.0018064145, 50),
        #0.10: np.linspace(0.0018619530, 0.001861745, 50),
        #0.11: np.linspace(0.0019203461, 0.0019205871, 50),
    },
    0.98: {
        0.0001: np.linspace(0.00141103,0.00141109,50),
        #0.0010: np.linspace(0.001414224,0.001414233,50), 
        #0.0001: np.linspace(0.00140880, 0.00141180, 50),
        #0.0010: np.linspace(0.001412787, 0.0014171895, 50),
        0.01: np.linspace(0.0014493108, 0.0014497432, 50),
        #0.02: np.linspace(0.0014870098, 0.0014874515, 50),
        #0.03: np.linspace(0.0015264395, 0.0015268823, 50),
        #0.04: np.linspace(0.001563747, 0.0015681492, 50),
        #0.05: np.linspace(0.0016109360, 0.0016113780, 50),
        #0.06: np.linspace(0.0016562653, 0.0016567070, 50),
        #0.07: np.linspace(0.0017038437, 0.0017042853, 50),
        #0.08: np.linspace(0.0017538362, 0.0017542775, 50),
        #0.09: np.linspace(0.001802463, 0.0018068645, 50),
        #0.10: np.linspace(0.0018618030, 0.001862245, 50),
        #0.11: np.linspace(0.0019201961, 0.0019206371, 50),
    },
}
"""
TEMPERATURE_SPECIFIC_FIELDS_BY_N = {
    0.50: {
        0.0001: np.linspace(0.0014145619, 0.0014145633, 50),
        0.0010: np.linspace(0.0014177508, 0.0014177515, 50),
    },
    0.95: {
        0.0001: np.linspace(0.00141395, 0.00141407, 50),
        0.0010: np.linspace(0.001417187, 0.0014171895, 50),
    },    
    0.96: {
        0.0001: np.linspace(0.00, 0.00, 2 ),
        0.0010: np.linspace(0.00, 0.00, 2),
    },
    0.98: {
        0.0001: np.linspace(0.00, 0.00, 2),
        0.0010: np.linspace(0.00, 0.00, 2),
    },
}
"""
# For backward compatibility, keep h_values_extended as a flat list
h_values_extended = np.linspace(0.0005, 0.0015, 30)  # Default range


# Magnetization analysis parameters - automatically derived from simulation parameters
# Select representative temperatures and band fillings for magnetization plots
# These ensure we have data for all cases since they're subsets of the simulation parameters
# You can modify these filters to change which cases are plotted
magnetization_temperatures = [T for T in temperatures_extended if T <= 8.0]  # Focus on low/medium T
magnetization_n_targets = [n for n in n_targets_extended if n >= 0.5]  # Focus on high filling

# Create magnetization cases from the simulation parameters
magnetization_cases = []
for T in magnetization_temperatures:
    for n_target in magnetization_n_targets:
        magnetization_cases.append({'T': T, 'n_target': n_target})

# Broken axis magnetization plot parameters
# Create consecutive temperature pairs for broken axis plots
# This will generate plots for: (0.0001, 0.001), (0.001, 0.01), (0.01, 0.1), (0.1, 1.0), (1.0, 8.0)
broken_axis_cases_pairs = []
temperatures_sorted = sorted(temperatures_extended)
for n_target in n_targets_extended:
    for i in range(len(temperatures_sorted) - 1):
        T1 = temperatures_sorted[i]
        T2 = temperatures_sorted[i + 1]
        broken_axis_cases_pairs.append([
            {'T': T1, 'n_target': n_target},
            {'T': T2, 'n_target': n_target}
        ])

print(f"Broken axis temperature pairs: {broken_axis_cases_pairs}")

# Keep the original broken_axis_cases for backward compatibility (first pair)
broken_axis_cases = broken_axis_cases_pairs[0] if broken_axis_cases_pairs else []

print(f"Magnetization analysis will generate plots for {len(magnetization_cases)} cases:")
print(f"  Temperatures: {magnetization_temperatures}")
print(f"  Band fillings: {magnetization_n_targets}")
print(f"  Broken axis cases: {broken_axis_cases}")

# Validate that broken axis cases exist in magnetization cases
for case in broken_axis_cases:
    if case not in magnetization_cases:
        print(f"  WARNING: Broken axis case {case} not found in magnetization_cases!")
        print(f"  This may cause missing data in broken axis plots.")

#temperatures_extended = [0.01]
#h_values_extended = [0.0]
#n_targets_extended = [0.8]

# Setup output directory structure for additional plots
output_dir_thermo = "quantum_limit_NFL_test_thermodynamics"
os.makedirs(output_dir_thermo, exist_ok=True)

# Create subdirectories for various analysis types
analysis_dirs = {}
analysis_types = ["by_temperature", "by_band_filling", "comparisons", "distributions"]
for analysis_type in analysis_types:
    analysis_dirs[analysis_type] = os.path.join(output_dir_thermo, analysis_type)
    os.makedirs(analysis_dirs[analysis_type], exist_ok=True)
    print(f"Created directory: {analysis_dirs[analysis_type]}")

# Collect data for comprehensive thermodynamic analysis
thermo_data = []
distribution_data = []  # Store distribution data for plotting
print("  - The 'shifting' is not a bug - it's the physics!")


def magnetization_brokenaxis_demo():
    """Run a small magnetization simulation and plot with professional broken axis formatting."""
    from chartformat_freeenergy_SPIN_ALMOST_Correct_Magnetization_Entropy3 import one_state
    from brokenaxes import brokenaxes
    from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import pandas as pd

    # Temperature and field ranges for broken axis plot
    temp_h_grid = {
        1e-4: np.linspace(0.00070720, 0.00070735, 50),
        1e-3: np.linspace(0.00070884, 0.00070888, 50),
    }
    
    # Inset data: wider h range for T=0.001 to show approach to n=0.99
    h_inset_range = np.linspace(0.00070, 0.00145, 50)

    distributions = ["FD", "FLB", "NFL"]
    distribution_colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
    n_target = 0.95
    n_target = 0.50
    #n_target = 0.96
    #n_target = 0.98
    Nk = 100
    t = -1.0
    tp = 0.25
    a = 1.0

    results = []
    for T, h_vals in temp_h_grid.items():
        kx = np.linspace(0, 2 * np.pi / a, Nk, endpoint=False)
        ky = kx.copy()
        for h in h_vals:
            for dist in distributions:
                row = one_state(dist, T, n_target, h, kx, ky, t, tp, a)
                M = row["n_up"] - row["n_down"]
                results.append({"T": T, "n_target": n_target, "f": dist, "h": h, "M": M})
                print(f"T = {T}, h = {h}, dist = {dist}, M = {M}")

    df = pd.DataFrame(results)
    temperatures = sorted(df["T"].unique())
    first_T = temperatures[0]
    
    # Create x-axis limits for broken axes (use 1e3 scaling like the example)
    x_limits = []
    for T in temperatures:
        t_group = df[df["T"] == T]
        h_values = t_group["h"].values * 1e4  # Use 1e3 like in the example
        x_limits.append((h_values.min(), h_values.max()))

    # Create figure and broken axes with proper spacing
    fig = plt.figure(figsize=(12, 8))
    bax = brokenaxes(xlims=x_limits, hspace=0.01, width_ratios=(1,1), wspace=0.02, fig=fig)

    # Plot data for each temperature
    for T in temperatures:
        t_group = df[df["T"] == T]
        
        # Plot each distribution
        for dist, dist_group in t_group.groupby("f"):
            if T == first_T:
                lbl = dist
            else:
                lbl = '_nolegend_'
            dist_group = dist_group.sort_values(by="h")
            bax.plot(dist_group["h"] * 1e4, dist_group["M"], 
                    marker='o', linestyle='-', 
                    label=lbl, color=distribution_colors[dist])

    # Custom tick formatter to show original h-values instead of scaled values
    def custom_h_formatter(x, pos):
        """Show full scaled numbers (like 7.00036) without offset notation."""
        # Show the full scaled number with appropriate precision
        if abs(x) < 0.01:
            return f"{x:.6f}"
        elif abs(x) < 0.1:
            return f"{x:.5f}"
        elif abs(x) < 1.0:
            return f"{x:.4f}"
        else:
            return f"{x:.3f}"
    
    # Set professional labels - showing scaled values with full numbers
    bax.set_xlabel(r"Zeeman field, $\mu_B H/|t| \times 10^{4}$", fontsize=20, labelpad=30)
    bax.set_ylabel(r"Magnetization, $M = (n_{\uparrow} - n_{\downarrow})$", fontsize=24, labelpad=50)
    
    # Add blue dashed horizontal line at n_target
    bax.axhline(n_target, linestyle='--', color='blue', linewidth=2, alpha=0.8)
    
    # Add legend without frame
    leg = bax.legend(fontsize=16, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.9))

    # Helper function to flatten axes
    def _flatten_axes(axs):
        if hasattr(axs, 'flatten'):
            return axs.flatten()
        elif hasattr(axs, 'values'):
            return list(axs.values())
        else:
            return axs

    # Professional tick formatting for each axis
    axes = _flatten_axes(bax.axs)
    for i, (T, ax) in enumerate(zip(temperatures, axes)):
        # Set tick locators
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(axis='both', labelsize=18)
        
        # Create subplot-specific formatter
        def make_formatter(subplot_idx):
            def formatter(x, pos):
                if subplot_idx == 1:  # Second subplot - one digit less
                    if abs(x) < 0.01:
                        return f"{x:.7f}"
                    elif abs(x) < 0.1:
                        return f"{x:.6f}"
                    elif abs(x) < 1.0:
                        return f"{x:.5f}"
                    else:
                        return f"{x:.4f}"
                else:  # First subplot - full precision
                    if abs(x) < 0.01:
                        return f"{x:.8f}"
                    elif abs(x) < 0.1:
                        return f"{x:.7f}"
                    elif abs(x) < 1.0:
                        return f"{x:.6f}"
                    else:
                        return f"{x:.5f}"
            return formatter
        
        # Apply custom formatter to x-axis to show full scaled numbers
        ax.xaxis.set_major_formatter(FuncFormatter(make_formatter(i)))
        
        # Scientific notation for x-axis
        #fmt = ScalarFormatter(useMathText=True, useOffset=False)
        #fmt.set_scientific(True)
        #fmt.set_powerlimits((0,0))
        #ax.xaxis.set_major_formatter(fmt)
        
        # Professional tick styling with selective tick placement
        ax.tick_params(
            which='major',
            direction='in',
            length=8,
            width=3,
            pad=6
        )
        
        # Control which ticks appear on each subplot to avoid middle ticks
        if i == 0:  # First subplot - no right ticks
            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
                bottom=True, top=True,
                left=True, right=False,
            )
        else:  # Second subplot - no left ticks
            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
                bottom=True, top=True,
                left=False, right=True,
            )
        
        # Control which spines appear on each subplot (analogical to ticks)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        if i == 0:  # First subplot - no right spine
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        else:  # Second subplot - no left spine
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
        
        # Temperature annotations - different positions for each subplot
        if i == 0:  # First subplot
            temp_format = -4
            x_pos = 0.95
            y_pos = 0.90
        else:  # Second subplot
            temp_format = -3
            x_pos = 0.55
            y_pos = 0.55
        
        label = rf"$\frac{{k_B T}}{{|t|}}=10^{{{temp_format}}}$"
        ax.text(x_pos, y_pos, label, transform=ax.transAxes,
                ha='right', va='center', fontsize=20)

    # Add n_target annotation on the last (rightmost) subplot
    ax = axes[0]  # Get the last axis
    ax.text(0.05, 0.91, rf"$n = {n_target:.2f}$", 
            transform=ax.transAxes,
            ha='left', va='center',
            color='blue', fontsize=18)

    # Add diagonal break lines at the top to match the bottom ones
    # Get the positions of the subplots
    pos_left = axes[0].get_position()
    pos_right = axes[1].get_position()
    
    # Calculate the gap between subplots
    gap_start = pos_left.x1  # Right edge of left subplot
    gap_end = pos_right.x0   # Left edge of right subplot
    gap_center = (gap_start + gap_end) / 2
    gap_width = gap_end - gap_start
    
    # Add diagonal break lines at the top
    top_y = pos_left.y1  # Top of the subplots
    line_height = 0.01  # Height of the diagonal lines
    
    # Left diagonal line (top)
    #fig.lines.append(plt.Line2D([gap_center - gap_width/4, gap_center], 
    #                           [top_y - line_height, top_y], 
    #                           color='black', linewidth=2, transform=fig.transFigure))
    # Right diagonal line (top)
    #fig.lines.append(plt.Line2D([gap_center, gap_center + gap_width/4], 
    #                           [top_y, top_y - line_height], 
    #                           color='black', linewidth=2, transform=fig.transFigure))

    # Create minimal inset plot showing wider h range for T=0.001
    # Smaller size, positioned left and down from upper right
    inset_ax = fig.add_axes([0.60, 0.60, 0.25, 0.20])  # [left, bottom, width, height] - moved left and down
    
    # Calculate inset data for all distributions at T=0.001
    T_inset = 1e-3  # 0.001
    kx_inset = np.linspace(0, 2 * np.pi / a, Nk, endpoint=False)
    ky_inset = kx_inset.copy()
    
    for dist in distributions:
        M_inset = []
        print(f"Computing inset data for {dist}...")
        
        for h_val in h_inset_range:
            state = one_state(dist, T_inset, n_target, h_val, kx_inset, ky_inset, t, tp, a)
            M_val = state["n_up"] - state["n_down"]
            M_inset.append(M_val)
            
        # Plot inset data - no markers, just lines
        inset_ax.plot(h_inset_range * 1e4, M_inset, '-', 
                     color=distribution_colors[dist], 
                     linewidth=2)
    
    # Add horizontal reference line at n_target
    inset_ax.axhline(n_target, linestyle='--', color='blue', linewidth=1.5, alpha=0.8)
    
    # Minimal formatting consistent with main plot
    # Remove grid, labels, title, legend
    inset_ax.tick_params(
        axis='both', 
        which='major',
        direction='in',
        length=3,  # Half the length of main plot ticks
        width=2,
        labelsize=12,
        pad=3
    )
    
    # Reduce number of ticks
    inset_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    inset_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    
    # Use plain formatting like the y-axis of main plot (no scientific notation)
    # Since we're already scaling by 1e4, the values will be in the same units as main plot
    inset_ax.ticklabel_format(style='plain', axis='both')
    
    # Make spines thicker and add distinctive border for inset
    for spine in inset_ax.spines.values():
        spine.set_linewidth(2.5)  # Thicker border to emphasize inset
        spine.set_edgecolor('black')
    
    # Tighten the inset layout by adjusting margins
    inset_ax.margins(x=0.05, y=0.05)  # Small margins for tighter fit
    
    # Add visual connection elements to show inset is extension of main plot
    # Find the rightmost point of the right subplot (where extension begins)
    right_ax = axes[-1]  # Last subplot
    
    # Get the x-axis limits of the rightmost subplot  
    x_right_max = right_ax.get_xlim()[1]  # Maximum x value of right subplot
    
    # Add connection lines from right edge of main plot to inset
    # Get positions in figure coordinates
    inset_pos = inset_ax.get_position()
    right_ax_pos = right_ax.get_position()
    
    # Connection line from top-left of right subplot to top-left of inset
    connection_line1 = plt.Line2D(
        [right_ax_pos.x0, inset_pos.x0], 
        [right_ax_pos.y1, inset_pos.y1],
        color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
        transform=fig.transFigure
    )
    fig.lines.append(connection_line1)
    
    # Connection line from bottom-left of right subplot to bottom-left of inset  
    connection_line2 = plt.Line2D(
        [right_ax_pos.x0, inset_pos.x0],
        [right_ax_pos.y0, inset_pos.y0], 
        color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
        transform=fig.transFigure
    )
    fig.lines.append(connection_line2)
    
    # Add diagonal extension lines from right corners of inset toward extended x-axis range
    # Calculate where the inset's max h value would be if main plot x-axis were extended
    h_inset_max = h_inset_range.max() * 1e4  # Maximum h value in inset (scaled)
    main_h_max = right_ax.get_xlim()[1]  # Current max h in main plot
    main_h_min = right_ax.get_xlim()[0]  # Current min h in main plot
    
    # Calculate how much further right the inset max would extend
    # Estimate the position as if the x-axis were extended proportionally
    extension_factor = (h_inset_max - main_h_min) / (main_h_max - main_h_min)
    extended_x_pos = right_ax_pos.x0 + (right_ax_pos.width * extension_factor)
    
    # Clamp the extended position to reasonable figure coordinates (prevent huge image)
    extended_x_pos = min(extended_x_pos, 0.98)  # Maximum 98% of figure width
    extended_x_pos = max(extended_x_pos, right_ax_pos.x1)  # At least right edge of main plot
    
    # Diagonal line from top-right of inset toward extended top axis
    diagonal_line1 = plt.Line2D(
        [inset_pos.x1, extended_x_pos],
        [inset_pos.y1, right_ax_pos.y1], 
        color='gray', linestyle=':', linewidth=1.2, alpha=0.6,
        transform=fig.transFigure
    )
    fig.lines.append(diagonal_line1)
    
    # Diagonal line from bottom-right of inset toward extended bottom axis
    diagonal_line2 = plt.Line2D(
        [inset_pos.x1, extended_x_pos],
        [inset_pos.y0, right_ax_pos.y0], 
        color='gray', linestyle=':', linewidth=1.2, alpha=0.6,
        transform=fig.transFigure
    )
    fig.lines.append(diagonal_line2)
    
    # Add annotation arrow pointing from inset to explain the extension
    plt.annotate('Extended h range\n(T = 10⁻³)', 
                xy=(0.75, 0.45),  # Point in figure coordinates
                xytext=(inset_pos.x0 + inset_pos.width/2, inset_pos.y0 - 0.05),  # Below inset
                ha='center', va='top',
                fontsize=10, color='gray',
                arrowprops=dict(arrowstyle='->', lw=1.2, color='gray', alpha=0.8),
                transform=fig.transFigure)

    # Save with high quality
    all_temps = list(temp_h_grid.keys())  # Convert dict_keys to list
    temp_string = "_".join([f"T{temp}" for temp in all_temps])  # Join all temperatures
    plt.savefig(f"magnetization_broken_axis_with_inset_Nk{Nk}_{temp_string}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved professional magnetization_broken_axis_with_inset_Nk{Nk}_{temp_string}.png")


#if __name__ == "__main__":
#    magnetization_brokenaxis_demo()

def collect_thermo_data_parallel(temperatures, h_values, n_targets, distributions, Nk, t, tp, a, t_up, tp_up, t_dn, tp_dn, n_cores=None):
    """
    Parallelized version of the main thermodynamic data collection loop.
    This is the main bottleneck and will see the highest speedup.
    """
    if n_cores is None:
        n_cores = mp.cpu_count()
    
    print(f"Starting parallel data collection with {n_cores} cores...")
    start_time = time.time()
    
    # Create all parameter combinations
    parameter_combinations = []
    for T in temperatures:
        for n_target in n_targets:
            try:
                h_range = TEMPERATURE_SPECIFIC_FIELDS_BY_N[n_target][T]
            except KeyError:
                raise KeyError(f"No h-range defined for n={n_target}, T={T}. Please add it to TEMPERATURE_SPECIFIC_FIELDS_BY_N.")
            print(f"  Using specific fields for T={T}, n={n_target}: {len(h_range)} points")
            for h in h_range:
                for dist in distributions:
                    parameter_combinations.append((dist, T, n_target, h, Nk, t, tp, a, t_up, tp_up, t_dn, tp_dn))
    
    total_combinations = len(parameter_combinations)
    print(f"Total calculations: {total_combinations}")
    
    # Process in parallel with progress tracking
    completed = 0
    thermo_data = []
    
    def update_progress(result):
        nonlocal completed
        completed += 1
        progress_interval = CONFIG['performance']['progress_update_interval']
        if completed % progress_interval == 0 or completed == total_combinations:  # Show progress every N or at the end
            progress = (completed / total_combinations) * 100
            elapsed = time.time() - start_time
            if completed > 0:
                eta = (elapsed / completed) * (total_combinations - completed)
                print(f"  Progress: {completed}/{total_combinations} ({progress:.1f}%) - ETA: {eta:.1f}s")
            else:
                print(f"  Progress: {completed}/{total_combinations} ({progress:.1f}%)")
    
    print(f"Starting calculations with {n_cores} parallel workers...")
    print("="*60)
    
    # Process in parallel using global worker function
    with mp.Pool(n_cores) as pool:
        for result in pool.imap_unordered(worker_thermo_data, parameter_combinations):
            if result is not None:
                thermo_data.append(result)
                # Show completion counter
                print(f"    ✓ Completed: {len(thermo_data)}/{total_combinations}")
            update_progress(result)
    
    end_time = time.time()
    print(f"Parallel data collection completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(thermo_data)} out of {total_combinations} combinations")
    
    # Add summary statistics
    if thermo_data:
        print(f"\nSummary:")
        print(f"  Total calculations: {total_combinations}")
        print(f"  Successful: {len(thermo_data)}")
        print(f"  Failed: {total_combinations - len(thermo_data)}")
        print(f"  Success rate: {(len(thermo_data)/total_combinations)*100:.1f}%")
    
    return thermo_data, distribution_data if 'distribution_data' in locals() else []


def solve_spin_split_state(dist, T, n_target, h, kx, ky, t_up, tp_up, t_dn, tp_dn, a):
    """Solve a spin-split state, optionally enforcing q_σ self-consistency."""
    sc_cfg = CONFIG.get('self_consistency', {})
    sc_enabled = sc_cfg.get('enabled', False)

    if sc_enabled:
        try:
            result = one_state_split_self_consistent(
                dist,
                T,
                n_target,
                h,
                kx,
                ky,
                t_up,
                tp_up,
                t_dn,
                tp_dn,
                a=a,
                max_iter=sc_cfg.get('max_iter', 50),
                tol=sc_cfg.get('tol', 1e-6),
                damping=sc_cfg.get('damping', 0.5),
                q_bounds=sc_cfg.get('q_bounds', (0.05, 5.0)),
            )
            if result.get('sc_converged', True) or not sc_cfg.get('fallback_on_failure', True):
                return result
            else:
                print(
                    "          Warning: self-consistent q_σ solver did not converge;"
                    " falling back to fixed hoppings."
                )
        except Exception as exc:
            if sc_cfg.get('fallback_on_failure', True):
                print(f"          Warning: self-consistent q_σ solver failed ({exc}); reverting to fixed hoppings.")
            else:
                raise

    base_result = one_state_split(dist, T, n_target, h, kx, ky, t_up, tp_up, t_dn, tp_dn, a)
    # Annotate with neutral self-consistency metadata for downstream consumers.
    base_result.update(
        {
            'q_up': 1.0,
            'q_down': 1.0,
            't_up_eff': t_up,
            'tp_up_eff': tp_up,
            't_down_eff': t_dn,
            'tp_down_eff': tp_dn,
            'sc_iterations': 0,
            'sc_converged': False if sc_enabled else None,
        }
    )
    return base_result

def worker_thermo_data(args):
    """
    Worker function for parallel thermodynamic data collection.
    Each worker processes one (distribution, T, n_target, h) combination.
    """
    dist, T, n_target, h, Nk, t, tp, a, t_up, tp_up, t_dn, tp_dn = args

    try:
        # Create k-space grid
        kx = np.linspace(0, 2 * np.pi, Nk, endpoint=False)
        ky = kx.copy()

        result = solve_spin_split_state(
            dist,
            T,
            n_target,
            h,
            kx,
            ky,
            t_up,
            tp_up,
            t_dn,
            tp_dn,
            a,
        )
        mu = result['mu']
        n_up = result['n_up']
        n_down = result['n_down']
        t_up_eff = result.get('t_up_eff', t_up)
        tp_up_eff = result.get('tp_up_eff', tp_up)
        t_dn_eff = result.get('t_down_eff', t_dn)
        tp_dn_eff = result.get('tp_down_eff', tp_dn)

        # Calculate magnetization M = n_up - n_down
        magnetization = n_up - n_down

        # Compute entropy using SEQUENTIAL version (not parallel to avoid daemonic process error)
        entropy = compute_total_entropy_split(
            T,
            h,
            mu,
            Nk,
            t_up_eff,
            tp_up_eff,
            t_dn_eff,
            tp_dn_eff,
            a,
            dist,
            n_target,
        )

        # Print detailed information for this completed calculation
        if CONFIG['performance']['detailed_output']:
            print(f"        Distribution: {dist}")
            print(f"          T = {T:.6f}, h = {h:.8f}, n = {n_target:.2f}, n_init = [{n_target/2:.2f},{n_target/2:.2f}]")
            print(f"          μ = {mu:.6f}, S = {entropy:.6f}, M = {magnetization:.6f}")
            print(f"          n_up = {n_up:.6f}, n_down = {n_down:.6f}")
            if result.get('sc_converged') is not None:
                status = '✓' if result.get('sc_converged') else '✗'
                print(
                    f"          q_up = {result.get('q_up', 1.0):.6f}, q_down = {result.get('q_down', 1.0):.6f}"
                    f" (self-consistent {status}, iterations = {result.get('sc_iterations', 0)})"
                )

        return {
            'f': dist,
            'T': T,
            '$n_{target}$': n_target,
            'h': h,
            'mu': mu,
            'n_up': n_up,
            'n_down': n_down,
            'M': magnetization,
            'S': entropy,
            'q_up': result.get('q_up'),
            'q_down': result.get('q_down'),
            't_up_eff': t_up_eff,
            'tp_up_eff': tp_up_eff,
            't_down_eff': t_dn_eff,
            'tp_down_eff': tp_dn_eff,
            'sc_converged': result.get('sc_converged'),
            'sc_iterations': result.get('sc_iterations'),
        }

    except Exception as e:
        print(f"Error processing {dist}, T={T}, n={n_target}, h={h}: {e}")
        return None

print("\nGenerating data for comprehensive thermodynamic analysis...")

# Choose between parallel and sequential processing
if USE_PARALLEL:
    print("Using PARALLEL processing for data collection...")
    thermo_data, distribution_data = collect_thermo_data_parallel(
        temperatures_extended, 
        h_values_extended, 
        n_targets_extended, 
        distributions, 
        Nk_test, 
        -1.0, 0.25, 1.0,
        t_up, tp_up, t_dn, tp_dn,
        N_CORES
    )
    
    # Also collect distribution data for plotting (using same parallel approach)
    print("\nCollecting distribution data for plotting...")
    distribution_data = []
    for result in thermo_data:
        if result is not None:
            # Create k-space grid for this result
            kx = np.linspace(0, 2 * np.pi, Nk_test, endpoint=False)
            ky = kx.copy()
            
            distribution_data.append({
                'f': result['f'],
                'T': result['T'],
                '$n_{target}$': result['$n_{target}$'],
                'h': result['h'],
                'mu': result['mu'],
                'kx': kx,
                'ky': ky,
            })
else:
    print("Using SEQUENTIAL processing for data collection...")
    thermo_data = []
    distribution_data = []
    total_calculations = len(temperatures_extended) * len(n_targets_extended) * len(distributions)
    current_calc = 0
    
    for T in temperatures_extended:
        print(f"  Temperature T = {T:.6f}")
        for n_target in n_targets_extended:
            try:
                h_range = TEMPERATURE_SPECIFIC_FIELDS_BY_N[n_target][T]
            except KeyError:
                raise KeyError(f"No h-range defined for n={n_target}, T={T}. Please add it to TEMPERATURE_SPECIFIC_FIELDS_BY_N.")
            print(f"    Using specific fields: {len(h_range)} points")
            for h in h_range:
                print(f"      Magnetic field h = {h:.6f}")
                for dist in distributions:
                    current_calc += 1
                    print(f"        Distribution: {dist} ({current_calc}/{total_calculations})")
                    
                    # Create k-space grid
                    kx = np.linspace(0, 2 * np.pi, Nk_test, endpoint=False)
                    ky = kx.copy()
                    
                    # Solve for this state
                    result = solve_spin_split_state(
                        dist,
                        T,
                        n_target,
                        h,
                        kx,
                        ky,
                        t_up,
                        tp_up,
                        t_dn,
                        tp_dn,
                        1.0,
                    )
                    mu = result['mu']
                    n_up = result['n_up']
                    n_down = result['n_down']
                    t_up_eff = result.get('t_up_eff', t_up)
                    tp_up_eff = result.get('tp_up_eff', tp_up)
                    t_dn_eff = result.get('t_down_eff', t_dn)
                    tp_dn_eff = result.get('tp_down_eff', tp_dn)

                    # Compute entropy
                    entropy = compute_total_entropy_split(
                        T,
                        h,
                        mu,
                        Nk_test,
                        t_up_eff,
                        tp_up_eff,
                        t_dn_eff,
                        tp_dn_eff,
                        1.0,
                        dist,
                        n_target,
                    )

                    # Store results
                    thermo_data.append({
                        'f': dist,
                        'T': T,
                        '$n_{target}$': n_target,
                        'h': h,
                        'mu': mu,
                        'n_up': n_up,
                        'n_down': n_down,
                        'S': entropy,
                        'M': n_up - n_down,  # Add magnetization calculation
                        'q_up': result.get('q_up'),
                        'q_down': result.get('q_down'),
                        't_up_eff': t_up_eff,
                        'tp_up_eff': tp_up_eff,
                        't_down_eff': t_dn_eff,
                        'tp_down_eff': tp_dn_eff,
                        'sc_converged': result.get('sc_converged'),
                        'sc_iterations': result.get('sc_iterations'),
                    })
                    
                    # Store distribution data for plotting
                    distribution_data.append({
                        'f': dist,
                        'T': T,
                        '$n_{target}$': n_target,
                        'h': h,
                        'mu': mu,
                        'kx': kx,
                        'ky': ky,
                    })
                    
                    print(f"          μ = {mu:.6f}, S = {entropy:.6f}")

# Create DataFrame from collected data
df_thermo = pd.DataFrame(thermo_data)

print("\nData collection completed. Generating plots...")

# Color and style settings
colors = {"FD": "blue", "FLB": "red", "NFL": "green"}
styles = {"FD": "-", "FLB": "--", "NFL": "-."}
labels = {"FD": "Fermi-Dirac", "FLB": "Fermi Liquid with Boltzmann", "NFL": "Non-Fermi Liquid"}
labels = {"FD": "FD", "FLB": "FLB", "NFL": "NFL"}

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

def format_temperature_annotation(T):
    """Format temperature for annotation with LaTeX scientific notation."""
    if T < 0.001:
        # Use scientific notation for very small temperatures
        exponent = int(np.floor(np.log10(abs(T))))
        mantissa = T / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            return rf"$\frac{{k_B T}}{{|t|}} = 10^{{{exponent}}}$"
        else:
            return rf"$\frac{{k_B T}}{{|t|}} = {mantissa:.1f} \cdot 10^{{{exponent}}}$"
    elif T < 0.01:
        # Use scientific notation for small temperatures
        exponent = int(np.floor(np.log10(abs(T))))
        mantissa = T / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            return rf"$\frac{{k_B T}}{{|t|}} = 10^{{{exponent}}}$"
        else:
            return rf"$\frac{{k_B T}}{{|t|}} = {mantissa:.1f} \cdot 10^{{{exponent}}}$"
    elif T < 0.1:
        # Use scientific notation for medium temperatures
        exponent = int(np.floor(np.log10(abs(T))))
        mantissa = T / (10 ** exponent)
        if abs(mantissa - 1.0) < 1e-10:
            return rf"$\frac{{k_B T}}{{|t|}} = 10^{{{exponent}}}$"
        else:
            return rf"$\frac{{k_B T}}{{|t|}} = {mantissa:.1f} \cdot 10^{{{exponent}}}$"
    else:
        # Use decimal notation for larger temperatures
        formatted = f"{T:.3f}".rstrip('0').rstrip('.')
        return rf"$\frac{{k_B T}}{{|t|}} = {formatted}$"

def add_plot_annotations(ax, T, h, n_target, x_pos=0.02, y_pos=0.02):
    """Add parameter annotations to plots instead of titles."""
    # Format temperature
    temp_text = format_temperature_annotation(T)
    
    # Format magnetic field with proper LaTeX fraction notation
    h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h)}$"
    
    # Format band filling
    n_text = f"$n = {n_target:.2f}$"
    
    # Combine annotations
    annotation_text = f"{temp_text}\n{h_text}\n{n_text}"
    
    ax.text(x_pos, y_pos, annotation_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='bottom', horizontalalignment='left')

def setup_plot_formatting(ax):
    """Apply consistent formatting to all plots"""
    ax.tick_params(axis='both', which='major', direction='in', length=8, width=1.5, labelsize=20)
    ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1,labelsize=20)
    ax.grid(True, alpha=0.3, linewidth=1)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.minorticks_on()

def get_optimal_legend_position(ax, annotation_pos=(0.02, 0.95)):
    """Determine optimal legend position to avoid overlap with annotations and data."""
    # Get the current data limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Check if annotation is in upper left (default)
    if annotation_pos[0] < 0.5 and annotation_pos[1] > 0.5:
        # Annotation is upper left, put legend in upper right
        return 'upper right'
    elif annotation_pos[0] > 0.5 and annotation_pos[1] > 0.5:
        # Annotation is upper right, put legend in upper left
        return 'upper left'
    elif annotation_pos[0] < 0.5 and annotation_pos[1] < 0.5:
        # Annotation is lower left, put legend in upper right
        return 'upper right'
    else:
        # Annotation is lower right, put legend in upper left
        return 'upper left'
"""
def get_dynamic_scaling_factor(h_values, temperature=None):
    # If temperature is provided, use temperature-specific scaling rules
    if temperature is not None:
        if temperature <= 0.001:  # T=0.0001 and T=0.001
            scale_factor = 1e4
            power = 4
        elif temperature <= 0.01:  # T=0.01
            scale_factor = 1e3
            power = 3
        elif temperature <= 0.1:  # T=0.1
            scale_factor = 1e2
            power = 2
        elif temperature <= 1.0:  # T=1.0
            scale_factor = 1e1
            power = 1
        else:  # T=8.0 and above
            scale_factor = 1.0
            power = 0
        return scale_factor, power
    
    # Fallback to magnitude-based scaling for general use
    h_avg = np.mean(np.abs(h_values))
    
    # Determine appropriate scaling based on the magnitude of h values
    if h_avg < 1e-6:
        # Very small values: scale by 1e8
        scale_factor = 1e8
        power = 8
    elif h_avg < 1e-5:
        # Very small values: scale by 1e7
        scale_factor = 1e7
        power = 7
    elif h_avg < 1e-4:
        # Very small values: scale by 1e6
        scale_factor = 1e6
        power = 6
    elif h_avg < 1e-3:
        # Small values: scale by 1e5
        scale_factor = 1e5
        power = 5
    elif h_avg < 1e-2:
        # Small values: scale by 1e4
        scale_factor = 1e4
        power = 4
    elif h_avg < 1e-1:
        # Medium values: scale by 1e3
        scale_factor = 1e3
        power = 3
    elif h_avg < 1.0:
        # Medium values: scale by 1e2
        scale_factor = 1e2
        power = 2
    elif h_avg < 10.0:
        # Large values: scale by 1e1
        scale_factor = 1e1
        power = 1
    else:
        # Very large values: no scaling
        scale_factor = 1.0
        power = 0
    
    return scale_factor, power
"""
def get_dynamic_scaling_factor(h_values, temperature=None, min_decades=None):
    """
    Determine scale_factor and power so that:
      scaled_h = h_values * scale_factor
    lives in roughly [1, 10] (or [0.1,1]) and the axis reads ×10^{power}.

    If min_decades is set, we'll guarantee at least that many decades
    of scaling (e.g. min_decades=2 for always pulling out 10^2 or more).
    """
    h_max = np.max(np.abs(h_values))
    if h_max == 0:
        return 1.0, 0

    # floor(log10) gives the decade of the max
    raw_pow = int(np.floor(np.log10(h_max)))
    # flip sign so small values (<1) yield positive powers
    power = -raw_pow

    # enforce minimum decades if requested
    if min_decades is not None:
        power = max(power, min_decades)

    scale_factor = 10**power
    return scale_factor, power

def get_dynamic_y_scaling(M_values, min_decades=None):
    """
    Determine y-axis scaling so that multiplied values show the first non-zero
    digit (mantissa in [1, 10)). Returns (scale_factor, power) with
    scaled_M = M_values * scale_factor and label multiplier ×10^{power}.
    """
    M_values = np.asarray(M_values)
    with np.errstate(invalid='ignore'):
        M_max = np.nanmax(np.abs(M_values))
    if not np.isfinite(M_max) or M_max == 0:
        return 1.0, 0
    raw_pow = int(np.floor(np.log10(M_max)))
    power = -raw_pow
    if min_decades is not None:
        power = max(power, min_decades)
    return 10**power, power

# 1. PLOTS GROUPED BY H VALUES (each h gets its own subfolder with n_target analysis)
print("Generating plots for each h value subfolder...")
# Collect all unique h values from actual data to avoid float mismatch
all_h_values = sorted(df_thermo['h'].unique())

for h_val in all_h_values:
    h_data = df_thermo[df_thermo['h'] == h_val]
    # Use very high precision for h to avoid folder name conflicts
    h_precision = 15
    
    h_dir = os.path.join(output_dir_thermo, f"h_{h_val:.{h_precision}f}")
    os.makedirs(h_dir, exist_ok=True)
    
    print(f"  Processing h = {h_val:.{h_precision}f} in {h_dir}")
    
    # For each n_target, create separate plots
    n_targets_present = sorted(h_data['$n_{target}$'].unique())
    for n_target in n_targets_present:
        n_data = h_data[h_data['$n_{target}$'] == n_target]
        
        # μ vs T for this (h, n_target) combination
        fig, ax = plt.subplots(figsize=(10, 8))
        for dist in distributions:
            dist_data = n_data[n_data['f'] == dist].sort_values('T')
            if not dist_data.empty:
                ax.plot(dist_data['T'], dist_data['mu'], 
                        color=colors[dist], linestyle=styles[dist], 
                        linewidth=3, marker='o', markersize=8, 
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=labels[dist])
        
        ax.set_xlabel('Temperature $k_B T/|t|$', fontsize=18)
        ax.set_ylabel('Chemical Potential $\\mu/|t|$', fontsize=18)
        # Remove title and add annotations - for μ vs T plots, show h and n_target (T is x-axis variable)
        if not n_data.empty:
            # For μ vs T plots, show h value and n_target since T is the x-axis variable
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{h_text}\n{n_text}"
            
            # Place legend in upper right, annotations just below it
            ax.legend(fontsize=14, frameon=False, loc='upper right')
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        else:
            # Fallback if no data - don't call legend() when there's no data
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{h_text}\n{n_text}"
            print(f"    No data to plot for h={h_val}, n={n_target}")
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        setup_plot_formatting(ax)
        plt.tight_layout()
        
        filename = os.path.join(h_dir, f"mu_vs_T_n{n_target:.2f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # S vs T for this (h, n_target) combination
        fig, ax = plt.subplots(figsize=(10, 8))
        for dist in distributions:
            dist_data = n_data[n_data['f'] == dist].sort_values('T')
            if not dist_data.empty:
                ax.plot(dist_data['T'], dist_data['S'], 
                        color=colors[dist], linestyle=styles[dist], 
                        linewidth=3, marker='o', markersize=8,
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=labels[dist])
        
        ax.set_xlabel('Temperature $k_B T/|t|$', fontsize=18)
        ax.set_ylabel('Entropy $S/k_B N$', fontsize=18)
        # Remove title and add annotations - for S vs T plots, show h and n_target (T is x-axis variable)
        if not n_data.empty:
            # For S vs T plots, show h value and n_target since T is the x-axis variable
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{h_text}\n{n_text}"
            
            # Place legend in upper right, annotations just below it
            ax.legend(fontsize=14, frameon=False, loc='upper right')
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        else:
            # Fallback if no data - don't call legend() when there's no data
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{h_text}\n{n_text}"
            print(f"    No data to plot for h={h_val}, n={n_target}")
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        setup_plot_formatting(ax)
        plt.tight_layout()
        
        filename = os.path.join(h_dir, f"S_vs_T_n{n_target:.2f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# 2. PLOTS GROUPED BY TEMPERATURE (μ vs h and S vs h for each n_target)
print("Generating plots grouped by temperature...")
for T_val in temperatures_extended:
    T_data = df_thermo[df_thermo['T'] == T_val]
    
    for n_target in n_targets_extended:
        n_data = T_data[T_data['$n_{target}$'] == n_target]
        
        # Get h-field range for this case and apply temperature-specific scaling
        h_values = n_data['h'].values
        scale_factor, power = get_dynamic_scaling_factor(h_values, temperature=T_val)
        
        # μ vs h for this (T, n_target) - DYNAMICALLY SCALED
        fig, ax = plt.subplots(figsize=(10, 8))
        for dist in distributions:
            dist_data = n_data[n_data['f'] == dist].sort_values('h')
            if not dist_data.empty:
                ax.plot(dist_data['h'] * scale_factor, dist_data['mu'], 
                        color=colors[dist], linestyle=styles[dist], 
                        linewidth=3, marker='o', markersize=8,
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=labels[dist])
        
        # Dynamic x-axis label based on scaling factor
        if power == 0:
            xlabel = 'Magnetic Field $\\mu_B H/|t|$'
        else:
            xlabel = f'Magnetic Field $\\mu_B H/|t| \\times 10^{{{power}}}$'
        
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel('Chemical Potential $\\mu/|t|$', fontsize=18)
        # Remove title and add annotations - for μ vs h plots, show T and n_target (h is x-axis variable)
        if not n_data.empty:
            # For μ vs h plots, show T and n_target since h is the x-axis variable
            temp_text = format_temperature_annotation(T_val)
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{temp_text}\n{n_text}"
            
            # Place legend in upper right, annotations just below it
            ax.legend(fontsize=14, frameon=False, loc='upper right')
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        else:
            # Fallback if no data - don't call legend() when there's no data
            temp_text = format_temperature_annotation(T_val)
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{temp_text}\n{n_text}"
            print(f"    No data to plot for T={T_val}, n={n_target}")
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        setup_plot_formatting(ax)
        plt.tight_layout()
        
        # Use higher precision for very small temperatures
        if T_val < 0.01:
            T_precision = 4  # 4 decimal places for very small temperatures
        else:
            T_precision = 3  # 3 decimal places for larger temperatures
        
        filename = os.path.join(analysis_dirs["by_temperature"], f"mu_vs_h_T{T_val:.{T_precision}f}_n{n_target:.2f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # S vs h for this (T, n_target) - DYNAMICALLY SCALED
        fig, ax = plt.subplots(figsize=(10, 8))
        for dist in distributions:
            dist_data = n_data[n_data['f'] == dist].sort_values('h')
            if not dist_data.empty:
                ax.plot(dist_data['h'] * scale_factor, dist_data['S'], 
                        color=colors[dist], linestyle=styles[dist], 
                        linewidth=3, marker='o', markersize=8,
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=labels[dist])
        
        # Dynamic x-axis label based on scaling factor (same as above)
        if power == 0:
            xlabel = 'Magnetic Field $\\mu_B H/|t|$'
        else:
            xlabel = f'Magnetic Field $\\mu_B H/|t| \\times 10^{{{power}}}$'
        
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel('Entropy $S/k_B N$', fontsize=18)
        # Remove title and add annotations - for S vs h plots, show T and n_target (h is x-axis variable)
        if not n_data.empty:
            # For S vs h plots, show T and n_target since h is the x-axis variable
            temp_text = format_temperature_annotation(T_val)
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{temp_text}\n{n_text}"
            
            # Place legend in upper right, annotations just below it
            ax.legend(fontsize=14, frameon=False, loc='upper right')
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        else:
            # Fallback if no data - don't call legend() when there's no data
            temp_text = format_temperature_annotation(T_val)
            n_text = f"$n = {n_target:.2f}$"
            annotation_text = f"{temp_text}\n{n_text}"
            print(f"    No data to plot for T={T_val}, n={n_target}")
            ax.text(0.98, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='right')
        setup_plot_formatting(ax)
        plt.tight_layout()
        
        filename = os.path.join(analysis_dirs["by_temperature"], f"S_vs_h_T{T_val:.{T_precision}f}_n{n_target:.2f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# 3. BAND FILLING ANALYSIS (μ vs n_target and S vs n_target)
print("Generating band filling analysis plots...")

# Debug: Show what data we actually have
print("Available data summary:")
for T_val in temperatures_extended:
    T_data = df_thermo[df_thermo['T'] == T_val]
    if not T_data.empty:
        h_vals = sorted(T_data['h'].unique())
        n_vals = sorted(T_data['$n_{target}$'].unique())
        print(f"  T={T_val}: {len(T_data)} points, h={h_vals}, n={n_vals}")
    else:
        print(f"  T={T_val}: No data")

for T_val in temperatures_extended:
    T_data = df_thermo[df_thermo['T'] == T_val]
    
    # Use h-values present in the data for this temperature to avoid float mismatches
    h_range = sorted(T_data['h'].unique())
    print(f"  Temperature T={T_val}: using {len(h_range)} h-values found in data")
    
    for h_val in h_range:
        h_data = T_data[T_data['h'] == h_val]
        
        # μ vs band filling for this (T, h)
        fig, ax = plt.subplots(figsize=(10, 8))
        for dist in distributions:
            dist_data = h_data[h_data['f'] == dist].sort_values('$n_{target}$')
            if not dist_data.empty:
                ax.plot(dist_data['$n_{target}$'], dist_data['mu'], 
                        color=colors[dist], linestyle=styles[dist], 
                        linewidth=3, marker='o', markersize=8,
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=labels[dist])
        
        ax.set_xlabel('Band Filling', fontsize=18)
        ax.set_ylabel('Chemical Potential $\\mu/|t|$', fontsize=18)
        # Remove title and add annotations - for μ vs n plots, show only T and h (n is x-axis variable)
        if not h_data.empty:
            # For μ vs n plots, show only T and h since n is the x-axis variable
            temp_text = format_temperature_annotation(T_val)
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            annotation_text = f"{temp_text}\n{h_text}"
            
            # Place legend in upper left, annotations just below it
            ax.legend(fontsize=14, frameon=False, loc='upper left')
            ax.text(0.02, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='left')
        else:
            # Fallback if no data - don't call legend() when there's no data
            temp_text = format_temperature_annotation(T_val)
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            annotation_text = f"{temp_text}\n{h_text}"
            print(f"    No data to plot for T={T_val}, h={h_val}")
            print(f"    Available h values for T={T_val}: {sorted(T_data['h'].unique())}")
            ax.text(0.02, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='left')
        setup_plot_formatting(ax)
        plt.tight_layout()
        
        # Use higher precision for very small temperatures
        if T_val < 0.01:
            T_precision = 4  # 4 decimal places for very small temperatures
        else:
            T_precision = 3  # 3 decimal places for larger temperatures
        
        # Use high precision for h values in filenames
        h_precision = 12
        
        filename = os.path.join(analysis_dirs["by_band_filling"], f"mu_vs_n_T{T_val:.{T_precision}f}_h{h_val:.{h_precision}f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # S vs band filling for this (T, h)
        fig, ax = plt.subplots(figsize=(10, 8))
        for dist in distributions:
            dist_data = h_data[h_data['f'] == dist].sort_values('$n_{target}$')
            if not dist_data.empty:
                ax.plot(dist_data['$n_{target}$'], dist_data['S'], 
                        color=colors[dist], linestyle=styles[dist], 
                        linewidth=3, marker='o', markersize=8,
                        markeredgewidth=1.5, markeredgecolor='white',
                        label=labels[dist])
        
        ax.set_xlabel('Band Filling', fontsize=18)
        ax.set_ylabel('Entropy $S/k_B N$', fontsize=18)
        # Remove title and add annotations - for S vs n plots, show only T and h (n is x-axis variable)
        if not h_data.empty:
            # For S vs n plots, show only T and h since n is the x-axis variable
            temp_text = format_temperature_annotation(T_val)
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            annotation_text = f"{temp_text}\n{h_text}"
            
            # Place legend in upper left, annotations just below it
            ax.legend(fontsize=14, frameon=False, loc='upper left')
            ax.text(0.02, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='left')
        else:
            # Fallback if no data - don't call legend() when there's no data
            temp_text = format_temperature_annotation(T_val)
            h_text = rf"$\frac{{h}}{{|t|}} = {format_magnetic_field(h_val)}$"
            annotation_text = f"{temp_text}\n{h_text}"
            print(f"    No data to plot for T={T_val}, h={h_val}")
            print(f"    Available h values for T={T_val}: {sorted(T_data['h'].unique())}")
            ax.text(0.02, 0.85, annotation_text, transform=ax.transAxes,
                    fontsize=14, verticalalignment='top', horizontalalignment='left')
        setup_plot_formatting(ax)
        plt.tight_layout()
        
        filename = os.path.join(analysis_dirs["by_band_filling"], f"S_vs_n_T{T_val:.{T_precision}f}_h{h_val:.{h_precision}f}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

# 4. MAGNETIZATION ANALYSIS (M vs h for specific T and n_target combinations)
print("Generating magnetization analysis plots...")

# Magnetization cases are now defined above using simulation parameters
# This ensures we have data for all cases since they're derived from temperatures_extended and n_targets_extended

# Create magnetization analysis directory
magnetization_dir = os.path.join(output_dir_thermo, "magnetization_analysis")
os.makedirs(magnetization_dir, exist_ok=True)

for case in magnetization_cases:
    T_val = case['T']
    n_target = case['n_target']
    
    # Filter data for this specific case
    case_data = df_thermo[(df_thermo['T'] == T_val) & (df_thermo['$n_{target}$'] == n_target)]
    
    if case_data.empty:
        print(f"  No data for T={T_val}, n={n_target}")
    # Get h-field range for this case and apply temperature-specific scaling
    h_values = case_data['h'].values
    scale_factor, power = get_dynamic_scaling_factor(h_values, temperature=T_val)
    
    print(f"    H-field range: {h_values.min():.2e} to {h_values.max():.2e}")
    print(f"    Using scaling factor: 10^{power}")
    print(f"    Data points: {len(case_data)} total")
    for dist in distributions:
        dist_data = case_data[case_data['f'] == dist]
        print(f"      {dist}: {len(dist_data)} points")
    
    # M vs h for this (T, n_target) combination - DYNAMICALLY SCALED
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for dist in distributions:
        dist_data = case_data[case_data['f'] == dist].sort_values('h')
        if not dist_data.empty:
            ax.plot(dist_data['h'] * scale_factor, dist_data['M'], 
                    color=colors[dist], linestyle=styles[dist], 
                    linewidth=3, marker='o', markersize=8,
                    markeredgewidth=1.5, markeredgecolor='white',
                    label=labels[dist])
    
    # Custom tick formatter to show full scaled numbers without offset notation
    def custom_h_formatter(x, pos):
        """Show full scaled numbers (like 7.00036) without offset notation."""
        # Show the full scaled number with appropriate precision
        if abs(x) < 0.01:
            return f"{x:.6f}"
        elif abs(x) < 0.1:
            return f"{x:.5f}"
        elif abs(x) < 1.0:
            return f"{x:.4f}"
        else:
            return f"{x:.3f}"
    
    # Set x-axis label to show scaled values with full numbers
    if power == 0:
        xlabel = 'Magnetic Field $\\mu_B H/|t|$'
    else:
        xlabel = f'Magnetic Field $\\mu_B H/|t| \\times 10^{{{power}}}$'
    
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel('Magnetization $M = n_{\\uparrow} - n_{\\downarrow}$', fontsize=22)
    
    # Add annotations
    temp_text = format_temperature_annotation(T_val)
    n_text = f"$n = {n_target:.2f}$"
    annotation_text = f"{temp_text}\n{n_text}"
    
    # Place legend and annotations
    ax.legend(fontsize=14, frameon=False, loc='upper left')
    ax.text(0.02, 0.85, annotation_text, transform=ax.transAxes,
            fontsize=14, verticalalignment='top', horizontalalignment='left')
    
    # Add horizontal reference line at n_target (maximum possible magnetization)
    ax.axhline(n_target, linestyle='--', color='blue', linewidth=2, alpha=0.8, 
              label=f'$n_{{target}} = {n_target:.2f}$')
    
    # Add zero line for reference
    ax.axhline(0, linestyle='-', color='black', linewidth=1, alpha=0.6)
    
    # Apply custom formatter to x-axis to show full scaled numbers
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(custom_h_formatter))
    
    # Configure y-axis scientific notation and offset position (left, closer to axis)
    from matplotlib.ticker import ScalarFormatter
    _yfmt = ScalarFormatter(useMathText=True)
    _yfmt.set_scientific(True)
    _yfmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(_yfmt)
    try:
        ax.yaxis.set_offset_position('left')
    except Exception:
        pass
    ax.yaxis.get_offset_text().set_x(-0.08)

    # Increase tick label sizes
    ax.tick_params(axis='both', labelsize=18)
    
    setup_plot_formatting(ax)
    plt.tight_layout()
    
    # Use higher precision for very small temperatures
    if T_val < 0.01:
        T_precision = 4  # 4 decimal places for very small temperatures
    else:
        T_precision = 3  # 3 decimal places for larger temperatures
    
    filename = os.path.join(magnetization_dir, f"M_vs_h_dynamic_scaled_T{T_val:.{T_precision}f}_n{n_target:.2f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a version with NO scaling for comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for dist in distributions:
        dist_data = case_data[case_data['f'] == dist].sort_values('h')
        if not dist_data.empty:
            ax.plot(dist_data['h'], dist_data['M'], 
                    color=colors[dist], linestyle=styles[dist], 
                    linewidth=3, marker='o', markersize=8,
                    markeredgewidth=1.5, markeredgecolor='white',
                    label=labels[dist])
    
    ax.set_xlabel('Magnetic Field $\\mu_B H/|t|$', fontsize=22)
    ax.set_ylabel('Magnetization $M = n_{\\uparrow} - n_{\\downarrow}$', fontsize=22)
    
    # Add annotations
    temp_text = format_temperature_annotation(T_val)
    n_text = f"$n = {n_target:.2f}$"
    annotation_text = f"{temp_text}\n{n_text}"
    
    # Place legend and annotations
    ax.legend(fontsize=16, frameon=False, loc='upper left')
    ax.text(0.02, 0.85, annotation_text, transform=ax.transAxes,
            fontsize=16, verticalalignment='top', horizontalalignment='left')
    
    # Add horizontal reference line at n_target
    ax.axhline(n_target, linestyle='--', color='blue', linewidth=2, alpha=0.8, 
              label=f'$n_{{target}} = {n_target:.2f}$')
    
    # Add zero line for reference
    ax.axhline(0, linestyle='-', color='black', linewidth=1, alpha=0.6)
    
    setup_plot_formatting(ax)
    plt.tight_layout()
    
    filename = os.path.join(magnetization_dir, f"M_vs_h_no_scaling_T{T_val:.{T_precision}f}_n{n_target:.2f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def format_temperature_annotation(T, sigfigs=2):
    """
    Return a LaTeX‐ready string of the form
      $k_B T/|t| = m \times 10^{p}$
    with m rounded to sigfigs significant figures.
    """
    if T == 0:
        return r"$\frac{k_B T}{|t|} = 0$"

    # find exponent p and mantissa m
    p = int(np.floor(np.log10(T)))
    m = T / (10**p)

    # round mantissa to the desired significant figures
    m_str = f"{m:.{sigfigs}g}"

    return rf"$\frac{{k_B T}}{{|t|}} = {m_str} \times 10^{{{p}}}$"

def create_broken_axis_magnetization_plot(df_thermo, output_dir, broken_axis_cases):
    """Create a broken axis magnetization plot using the collected thermodynamic data."""
    try:
        from brokenaxes import brokenaxes
        from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter
        
        # Filter data for these cases
        plot_data = []
        temperatures = []
        for case in broken_axis_cases:
            T_val = case['T']
            n_target = case['n_target']
            case_data = df_thermo[(df_thermo['T'] == T_val) & (df_thermo['$n_{target}$'] == n_target)]
            if not case_data.empty:
                plot_data.append(case_data)
                temperatures.append(T_val)
            else:
                print(f"    Warning: No data found for T={T_val}, n={n_target}")
        
        if len(plot_data) < 2:
            print("  Not enough data for broken axis plot")
            return
        
        # Create x-axis limits for broken axes with consistent scaling
        # Use a single scaling factor for all subplots to maintain consistency
        x_limits = []
        
        # Gather all h-values from all subplots to determine a single scaling factor
        all_h_values = np.concatenate([case_data['h'].values for case_data in plot_data])
        all_temperatures = [case_data['T'].iloc[0] for case_data in plot_data]
        
        # For broken axis plots, use the most common scaling factor
        # If temperatures have different scaling, use the scaling of the first temperature
        primary_T = all_temperatures[0]
        scale_factor, power = get_dynamic_scaling_factor(all_h_values, temperature=primary_T)
        
        print(f"    Using consistent scaling: ×10^{power} for all subplots")
        
        for case_data in plot_data:
            h_values = case_data['h'].values
            h_values_scaled = h_values * scale_factor  # Use same scaling for all
            x_limits.append((h_values_scaled.min(), h_values_scaled.max()))
        
        # Create figure and broken axes
        fig = plt.figure(figsize=(12, 8))
        bax = brokenaxes(xlims=x_limits, hspace=0.01, width_ratios=(1,1), wspace=0.02, fig=fig)
        # Dynamic y scaling from all magnetization values across both subplots
        try:
            all_M_vals = np.concatenate([cd['M'].values for cd in plot_data])
        except Exception:
            all_M_vals = np.array([])
        y_scale_factor, y_power = get_dynamic_y_scaling(all_M_vals)
        
        # Plot data for each temperature (using consistent scaling)
        for i, (T_val, case_data) in enumerate(zip(temperatures, plot_data)):
            # Plot each distribution
            for dist in distributions:
                dist_data = case_data[case_data['f'] == dist].sort_values('h')
                if not dist_data.empty:
                    if i == 0:  # First temperature gets legend
                        lbl = labels[dist]
                    else:
                        lbl = '_nolegend_'
                    
                    print(f"      {dist}: {len(dist_data)} points, h range: {dist_data['h'].min():.2e} to {dist_data['h'].max():.2e}")
                    bax.plot(dist_data['h'] * scale_factor, dist_data['M'] * y_scale_factor, 
                            marker='o', linestyle='-', 
                            label=lbl, color=colors[dist])
        
        # Set professional labels - showing scaled values with full numbers
        if power == 0:
            xlabel = r"zeeman field, $\mu_B H/|t|$"
        else:
            xlabel = rf"zeeman field, $\mu_B H/|t| \times 10^{{{power}}}$"
        
        bax.set_xlabel(xlabel, fontsize=24, labelpad=30)
        # Plain y ticks; multiplier only in label
        from matplotlib.ticker import ScalarFormatter as _SF
        _axs_plain = bax.axs if hasattr(bax.axs, '__iter__') else [bax.axs]
        for _ax in _axs_plain:
            _yfmt = _SF(useMathText=True)
            _yfmt.set_scientific(False)
            _yfmt.set_useOffset(False)
            _ax.yaxis.set_major_formatter(_yfmt)
            try:
                _ax.yaxis.get_offset_text().set_visible(False)
            except Exception:
                pass
        base_label = r"magnetic moment, $m = (n_{\uparrow} - n_{\downarrow})$"
        if y_power == 0:
            bax.set_ylabel(base_label, fontsize=24, labelpad=50)
        else:
            bax.set_ylabel(base_label + rf" $\times 10^{{{y_power}}}$", fontsize=24, labelpad=50)
        
        # Add legend only if there's data to plot
        if any(len(case_data[case_data['f'] == dist]) > 0 for case_data in plot_data for dist in distributions):
            bax.legend(fontsize=18, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.9))
        
        # Custom tick formatter to show full scaled numbers without offset notation
        def custom_h_formatter(x, pos, subplot_index=0):
            """Show full scaled numbers (like 7.00036) without offset notation."""
            # Show the full scaled number with appropriate precision (2 more decimal places)
            # Second subplot (index 1) shows one digit less
            if broken_axis_cases[0]['n_target'] ==0.5:
                return f"{x:.7f}"
            if subplot_index == 1:  # Second subplot - one digit less
                if abs(x) < 0.01:
                    return f"{x:.7f}"
                elif abs(x) < 0.1:
                    return f"{x:.6f}"
                elif abs(x) < 1.0:
                    return f"{x:.5f}"
                else:
                    return f"{x:.5f}"
            else:  # First subplot - full precision
                if abs(x) < 0.01:
                    return f"{x:.8f}"
                elif abs(x) < 0.1:
                    return f"{x:.7f}"
                elif abs(x) < 1.0:
                    return f"{x:.6f}"
                else:
                    return f"{x:.6f}"
        old_T=0
        # Professional formatting for each subplot
        axes = bax.axs if hasattr(bax.axs, '__iter__') else [bax.axs]
        for i, (T_val, ax) in enumerate(zip(temperatures, axes)):
            # Set tick locators
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis='both', labelsize=22)
            # Create subplot-specific formatter
            def make_formatter(subplot_idx):
                def formatter(x, pos):
                    if broken_axis_cases[0]['n_target'] ==0.5:
                            return f"{x:.7f}"
                    if subplot_idx == 1:  # Second subplot - one digit less
                        if abs(x) < 0.01:
                            return f"{x:.7f}"
                        elif abs(x) < 0.1:
                            return f"{x:.6f}"
                        elif abs(x) < 1.0:
                            return f"{x:.5f}"
                        else:
                            return f"{x:.6f}"
                    else:  # First subplot - full precision
                        if abs(x) < 0.00001:
                            return f"{x:.8f}"
                        elif abs(x) < 0.1:
                            return f"{x:.7f}"
                        elif abs(x) < 1.0:
                            return f"{x:.6f}"
                        else:
                            return f"{x:.6f}"
                return formatter
            
            # Apply custom formatter to x-axis to show full scaled numbers
            ax.xaxis.set_major_formatter(FuncFormatter(make_formatter(i)))
            
            # Professional tick styling
            ax.tick_params(which='major', direction='in', length=8, width=3, pad=6)
            
            # Control which ticks appear on each subplot
            if i == 0:  # First subplot - no right ticks
                ax.tick_params(axis='both', which='both', direction='in',
                             bottom=True, top=True, left=True, right=False)
            else:  # Second subplot - no left ticks
                ax.tick_params(axis='both', which='both', direction='in',
                             bottom=True, top=True, left=False, right=True)
            
            # Control spines
            for spine in ax.spines.values():
                spine.set_linewidth(2)
                spine.set_zorder(5)
            temp_label = format_temperature_annotation(T_val)
            ax.text(0.97, 0.15, temp_label, transform=ax.transAxes, ha='right', va='center', fontsize=24)
        #plt.tight_layout()  
        # Add n_target annotation
        axs = bax.axs if hasattr(bax.axs, '__iter__') else [bax.axs]

        n_target = broken_axis_cases[0]['n_target']
        ax = axes[0]
        ax.text(0.05, 0.66, rf"$n = {n_target:.2f}$", 
                transform=ax.transAxes,
                ha='left', va='center',
                color='blue', fontsize=18)
        # y ticks remain plain; label already includes ×10^{y_power}
        # Save the plot
        filename = os.path.join(output_dir, "magnetization_broken_axis_from_data.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Broken axis magnetization plot saved: {filename}")
        
    except ImportError:
        print("  Warning: brokenaxes not available, skipping broken axis plot")
    except Exception as e:
        print(f"  Warning: Could not create broken axis plot: {e}")

# Helper: compute a single inset M value (top-level for safe multiprocessing)
def _compute_inset_M_single(args):
    dist, T_val, n_target, h_val, Nk_inset, t_up, tp_up, t_dn, tp_dn, a = args
    try:
        kx = np.linspace(0, 2 * np.pi, Nk_inset, endpoint=False)
        ky = kx.copy()
        res = solve_spin_split_state(
            dist,
            T_val,
            n_target,
            h_val,
            kx,
            ky,
            t_up,
            tp_up,
            t_dn,
            tp_dn,
            a,
        )
        return res['n_up'] - res['n_down']
    except Exception:
        return np.nan


def create_broken_axis_magnetization_plot_with_inset(
    df_thermo,
    output_dir,
    broken_axis_cases,
    inset_h_max=0.004,
    inset_samples=inset_samples,
    Nk_inset=Nk_inset,
    t=-1.0,
    tp=0.25,
    t_up=None,
    tp_up=None,
    t_dn=None,
    tp_dn=None,
    a=1.0,
    inset_distributions=None,
    inset_position=(0.66, 0.62, 0.22, 0.18),
    use_parallel_inset=False,
    max_workers=None,
    inset_T_only=0.01
):
    """
    Extended version of create_broken_axis_magnetization_plot that adds an inset
    showing M(h) on a wider range [0, inset_h_max].
    broken_axis_cases: list of two dicts [{'T': T1, 'n_target': n}, {'T': T2, 'n_target': n}]
    """
    try:
        from brokenaxes import brokenaxes
        from matplotlib.ticker import MaxNLocator, ScalarFormatter, FuncFormatter

        if t_up is None:
            t_up = t
        if tp_up is None:
            tp_up = tp
        if t_dn is None:
            t_dn = t_up
        if tp_dn is None:
            tp_dn = tp_up

        # Filter data for these cases
        plot_data = []
        temperatures = []
        for case in broken_axis_cases:
            T_val = case['T']
            n_target = case['n_target']
            case_data = df_thermo[(df_thermo['T'] == T_val) & (df_thermo['$n_{target}$'] == n_target)]
            if not case_data.empty:
                plot_data.append(case_data)
                temperatures.append(T_val)
            else:
                print(f"    Warning: No data found for T={T_val}, n={n_target}")
        if len(plot_data) < 2:
            print("  Not enough data for broken axis plot (with inset)")
            return

        # Consistent scaling across subplots using first temperature
        all_h_values = np.concatenate([case_data['h'].values for case_data in plot_data])
        primary_T = temperatures[0]
        scale_factor, power = get_dynamic_scaling_factor(all_h_values, temperature=primary_T)
        print(f"    Using consistent scaling (inset): ×10^{power} for all subplots")

        x_limits = []
        for case_data in plot_data:
            h_scaled = case_data['h'].values * scale_factor
            x_limits.append((h_scaled.min(), h_scaled.max()))

        # Create figure and broken axes
        fig = plt.figure(figsize=(12, 8))
        bax = brokenaxes(xlims=x_limits, hspace=0.01, width_ratios=(1, 1), wspace=0.02, fig=fig)
        # Dynamic y scaling from all M values across both subplots
        try:
            all_M_vals = np.concatenate([cd['M'].values for cd in plot_data])
        except Exception:
            all_M_vals = np.array([])
        y_scale_factor, y_power = get_dynamic_y_scaling(all_M_vals)

        # Plot data
        for i, (T_val, case_data) in enumerate(zip(temperatures, plot_data)):
            for dist in distributions:
                dist_data = case_data[case_data['f'] == dist].sort_values('h')
                if not dist_data.empty:
                    lbl = labels[dist] if i == 0 else '_nolegend_'
                    bax.plot(dist_data['h'] * scale_factor, dist_data['M'] * y_scale_factor,
                             marker='o', linestyle='-', label=lbl, color=colors[dist])

        # Labels
        if power == 0:
            xlabel = r"zeeman field, $\mu_B H/|t|$"
        else:
            xlabel = rf"zeeman field, $\mu_B H/|t| \times 10^{{{power}}}$"
        bax.set_xlabel(xlabel, fontsize=24, labelpad=30)
        # Plain y ticks with multiplier only in label
        from matplotlib.ticker import ScalarFormatter as _SF
        axes_plain = bax.axs if hasattr(bax.axs, '__iter__') else [bax.axs]
        for _ax in axes_plain:
            _yfmt = _SF(useMathText=True)
            _yfmt.set_scientific(False)
            _yfmt.set_useOffset(False)
            _ax.yaxis.set_major_formatter(_yfmt)
            try:
                _ax.yaxis.get_offset_text().set_visible(False)
            except Exception:
                pass
        base_label = r"magnetic moment, $m = (n_{\uparrow} - n_{\downarrow})$"
        if y_power == 0:
            bax.set_ylabel(base_label, fontsize=24, labelpad=50)
        else:
            bax.set_ylabel(base_label + rf" $\times 10^{{{y_power}}}$", fontsize=24, labelpad=50)

        # Legend
        if any(len(cd[cd['f'] == d]) > 0 for cd in plot_data for d in distributions):
            bax.legend(fontsize=18, loc='upper left', frameon=False, bbox_to_anchor=(0.0, 0.9))

        # Tick formatting
        def make_formatter(subplot_idx):
            def formatter(x, pos):
                if subplot_idx == 1:
                    if abs(x) < 0.01:
                        return f"{x:.7f}"
                    elif abs(x) < 0.1:
                        return f"{x:.6f}"
                    elif abs(x) < 1.0:
                        return f"{x:.6f}"
                    else:
                        return f"{x:.6f}"
                else:
                    if abs(x) < 0.01:
                        return f"{x:.8f}"
                    elif abs(x) < 0.1:
                        return f"{x:.7f}"
                    elif abs(x) < 1.0:
                        return f"{x:.6f}"
                    else:
                        return f"{x:.6f}"
            return formatter

        axes = bax.axs if hasattr(bax.axs, '__iter__') else [bax.axs]
        for i, (T_val, ax) in enumerate(zip(temperatures, axes)):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune='both'))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.tick_params(axis='both', labelsize=22)
            ax.xaxis.set_major_formatter(FuncFormatter(make_formatter(i)))
            ax.tick_params(which='major', direction='in', length=8, width=3, pad=6)
            if i == 0:
                ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=False)
            else:
                ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=False, right=True)
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            temp_label = format_temperature_annotation(T_val)
            ax.text(0.97, 0.15, temp_label, transform=ax.transAxes, ha='right', va='center', fontsize=24)

        # n annotation
        n_target = broken_axis_cases[0]['n_target']
        axes[0].text(0.05, 0.66, rf"$n = {n_target:.2f}$", transform=axes[0].transAxes,
                     ha='left', va='center', color='blue', fontsize=18)
        # y ticks already plain above and label has ×10^{power}

        # Inset M(h) on [0, inset_h_max]
        if inset_distributions is None:
            inset_distributions = distributions
        try:
            print(f"    Inset: temperatures={temperatures}, distributions={list(inset_distributions)}")
        except Exception:
            pass
        inset_ax = fig.add_axes(inset_position)
        # Make inset background transparent and behind outer spines
        inset_ax.set_facecolor('none')
        try:
            inset_ax.patch.set_alpha(0.0)
        except Exception:
            pass
        inset_ax.set_zorder(1)
        h_inset = np.linspace(0.0, inset_h_max, inset_samples)
        print(f"    Inset: h grid [{h_inset[0]:.6g}, {h_inset[-1]:.6g}]")
        # Only plot for specified temperature
        inset_temperatures = [t for t in temperatures if abs(t - inset_T_only) < 1e-12]
        for T_val in inset_temperatures:
            for dist in inset_distributions:
                if use_parallel_inset:
                    try:
                        from concurrent.futures import ProcessPoolExecutor, as_completed
                        workers = max_workers
                        print(f"      Inset: parallel mode (workers={workers if workers else 'auto'}) for T={T_val:.6g}, dist={dist}")
                        args_iter = [
                            (
                                dist,
                                T_val,
                                n_target,
                                h_val,
                                Nk_inset,
                                t_up,
                                tp_up,
                                t_dn,
                                tp_dn,
                                a,
                            )
                            for h_val in h_inset
                        ]
                        M_inset = [np.nan] * len(h_inset)
                        with ProcessPoolExecutor(max_workers=workers) as ex:
                            futures = {ex.submit(_compute_inset_M_single, args): idx for idx, args in enumerate(args_iter)}
                            done = 0
                            report_every = max(1, len(futures)//10)
                            for fut in as_completed(futures):
                                idx = futures[fut]
                                try:
                                    M_inset[idx] = fut.result()
                                except Exception:
                                    M_inset[idx] = np.nan
                                done += 1
                                if done % report_every == 0 or done == len(futures):
                                    print(f"        Inset progress (T={T_val:.6g}, {dist}): {done}/{len(futures)}")
                    except Exception as e:
                        print(f"      Inset: parallel failed ({e}); falling back to sequential for T={T_val:.6g}, dist={dist}")
                        use_parallel_inset = False
                if not use_parallel_inset:
                    M_inset = []
                    total = len(h_inset)
                    for i_idx, h_val in enumerate(h_inset, start=1):
                        try:
                            kx = np.linspace(0, 2 * np.pi, Nk_inset, endpoint=False)
                            ky = kx.copy()
                            res = solve_spin_split_state(
                                dist,
                                T_val,
                                n_target,
                                h_val,
                                kx,
                                ky,
                                t_up,
                                tp_up,
                                t_dn,
                                tp_dn,
                                a,
                            )
                            M_inset.append(res['n_up'] - res['n_down'])
                        except Exception:
                            M_inset.append(np.nan)
                        if i_idx % max(1, total//10) == 0 or i_idx == total:
                            print(f"        Inset progress (T={T_val:.6g}, {dist}): {i_idx}/{total}")
                M_arr = np.array(M_inset)
                try:
                    mmin = np.nanmin(M_arr); mmax = np.nanmax(M_arr)
                except Exception:
                    mmin = np.nan; mmax = np.nan
                finite = int(np.isfinite(M_arr).sum())
                print(f"      Inset: T={T_val:.6g}, dist={dist}: finite={finite}/{len(M_arr)}, M in [{mmin:.6g}, {mmax:.6g}]")
                inset_ax.plot(h_inset, M_arr, '-', lw=1.6,
                               label=None,
                               color=colors.get(dist, 'black'))
        inset_ax.set_xlim(0.0, inset_h_max)
        inset_ax.set_xlabel('$\mu_B H/|t|$', fontsize=9)
        inset_ax.set_ylabel('$m$', fontsize=9)
        inset_ax.tick_params(which='both', direction='in', length=4, width=1.0, pad=2, labelsize=8)
        # Remove legend on inset
        # inset_ax.legend(fontsize=8, frameon=False, loc='upper left')  # removed
        inset_ax.axhline(n_target, linestyle='--', color='blue', linewidth=1.0, alpha=0.8)
        inset_ax.ticklabel_format(style='plain', axis='both')
        # Ensure inset spines are visible and thinner
        for side in ['left', 'right', 'top', 'bottom']:
            inset_ax.spines[side].set_visible(True)
            inset_ax.spines[side].set_linewidth(1.2)
        # Add temperature annotation to inset (same style as other annotations)
        try:
            if inset_temperatures:
                temp_label_inset = format_temperature_annotation(inset_temperatures[0])
                inset_ax.text(0.98, 0.75, temp_label_inset, transform=inset_ax.transAxes,
                              ha='right', va='top', fontsize=9)
        except Exception:
            pass

        # Draw explicit outer top and right frame lines to ensure visibility
        try:
            from matplotlib.lines import Line2D
            # Compute combined bbox of the broken axes
            ax_positions = [ax.get_position() for ax in axes]
            x0 = min(p.x0 for p in ax_positions)
            x1 = max(p.x1 for p in ax_positions)
            y0 = min(p.y0 for p in ax_positions)
            y1 = max(p.y1 for p in ax_positions)
            # Top line
            fig.add_artist(Line2D([x0, x1], [y1, y1], transform=fig.transFigure,
                                  color='black', linewidth=2.0, zorder=10, clip_on=False))
            # Right line
            fig.add_artist(Line2D([x1, x1], [y0, y1], transform=fig.transFigure,
                                  color='black', linewidth=2.0, zorder=10, clip_on=False))
        except Exception:
            pass

        # Save
        try:
            # Final guard: enforce plain y ticks on all axes (main + inset)
            fig.canvas.draw()
            from matplotlib.ticker import ScalarFormatter as _ScalarFormatterNoSci
            for _ax in fig.get_axes():
                try:
                    _yf = _ScalarFormatterNoSci(useMathText=True)
                    _yf.set_scientific(False)
                    _yf.set_useOffset(False)
                    _ax.yaxis.set_major_formatter(_yf)
                    _ax.ticklabel_format(axis='y', style='plain')
                    _ax.yaxis.get_offset_text().set_visible(False)
                except Exception:
                    pass
        except Exception:
            pass
        filename = os.path.join(output_dir, "magnetization_broken_axis_with_inset.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Broken axis magnetization plot with inset saved: {filename}")

    except ImportError:
        print("  Warning: brokenaxes not available, skipping broken axis plot with inset")
    except Exception as e:
        print(f"  Warning: Could not create broken axis plot with inset: {e}")

# Create broken axis plots for all consecutive temperature pairs
def create_all_broken_axis_plots(df_thermo, output_dir, broken_axis_cases_pairs):
    """Create broken axis magnetization plots for all consecutive temperature pairs."""
    print(f"  Creating broken axis plots for {len(broken_axis_cases_pairs)} temperature pairs...")
    
    for i, temperature_pair in enumerate(broken_axis_cases_pairs):
        T1, T2 = temperature_pair[0]['T'], temperature_pair[1]['T']
        n_target = temperature_pair[0]['n_target']  # Should be the same for both
        
        print(f"    Creating broken axis plot for T={T1} and T={T2}, n={n_target}")
        
        # Create broken axis plot for this pair
        # create_broken_axis_magnetization_plot(df_thermo, output_dir, temperature_pair)
        create_broken_axis_magnetization_plot_with_inset(
            df_thermo,
            output_dir,
            temperature_pair,
            use_parallel_inset=True,
            max_workers=32,
            inset_T_only=T2,
            t=t,
            tp=tp,
            t_up=t_up,
            tp_up=tp_up,
            t_dn=t_dn,
            tp_dn=tp_dn,
        )
        
        # Rename the file to include temperature pair and n info
        #old_filename = os.path.join(output_dir, "magnetization_broken_axis_from_data.png")
        old_filename = os.path.join(output_dir, "magnetization_broken_axis_with_inset.png")
        new_filename = os.path.join(output_dir, f"magnetization_broken_axis_T{T1:.4f}_T{T2:.4f}_n{n_target:.2f}.png")
        
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            print(f"      Saved: {new_filename}")
        else:
            print(f"      Warning: Could not create broken axis plot for T={T1}, T={T2}, n={n_target}")

# Create all broken axis plots
create_all_broken_axis_plots(df_thermo, magnetization_dir, broken_axis_cases_pairs)

# Note: Removed redundant call to create_broken_axis_magnetization_plot() 
# since create_all_broken_axis_plots() already creates plots for all temperature pairs


# 5. DOS
print("Generating DOS plots...")

def plot_dos_with_mu(
    mu,
    kx=None,
    ky=None,
    *,
    Nk=100,
    t=-1.0,
    tp=0.25,
    a=1.0,
    h=0.0,
    spin_resolved=True,
    bins=600,
    energy_window=None,
    gaussian_sigma=None,       # optional Gaussian width (energy units)
    lorentzian_eta=None,       # optional Lorentzian half-width at half-maximum (energy units)
    n_k_shifts=1,              # average over randomly shifted k-meshes
    shift_seed=None,           # RNG seed for reproducibility
    normalize=True,
    fill_occupied=True,
    ax=None,
    colors=("tab:blue", "tab:orange"),
    label_prefix="",
    output_path=None,
    annotate_kgrid=True,
    separate_plots=False,
):
    """
    Plot DOS g(E) using a k-grid histogram of the tight-binding dispersion.
    μ is GIVEN (no recalculation). Optional Lorentzian/Gaussian broadening and
    multi–shift k-mesh averaging to reduce jaggedness.

    Returns
    -------
    E_centers, g_up, g_dn, g_tot, ax
    """

    # -------------------- k-grids (possibly multi-shift) --------------------
    rng = np.random.default_rng(shift_seed)

    def _make_mesh(shift_x=0.0, shift_y=0.0):
        if kx is None or ky is None:
            # uniform grid in [0, 2π/a)
            kx_loc = np.linspace(0, 2*np.pi/a, Nk, endpoint=False)
            ky_loc = kx_loc.copy()
        else:
            kx_loc, ky_loc = np.array(kx, copy=True), np.array(ky, copy=True)

        # apply fractional shift in units of a single k-step
        if shift_x or shift_y:
            if len(kx_loc) > 1:
                dk = (kx_loc[1] - kx_loc[0])
            else:
                dk = 0.0
            kx_loc = (kx_loc + shift_x * dk) % (2*np.pi/a)
            ky_loc = (ky_loc + shift_y * dk) % (2*np.pi/a)
        KX, KY = np.meshgrid(kx_loc, ky_loc, indexing="ij")
        return KX, KY

    # -------------------- energies (spin split if h≠0) ---------------------
    def _collect_energies(KX, KY):
        Ek = tight_binding_dispersion(KX, KY, t, tp, a).ravel()
        if spin_resolved:
            return Ek - 0.5*h, Ek + 0.5*h
        else:
            return Ek, None

    # Make first mesh (no shift) to set energy window
    KX0, KY0 = _make_mesh(0.0, 0.0)
    Ek_up0, Ek_dn0 = _collect_energies(KX0, KY0)
    if spin_resolved:
        Emin = float(min(Ek_up0.min(), Ek_dn0.min()))
        Emax = float(max(Ek_up0.max(), Ek_dn0.max()))
    else:
        Emin = float(Ek_up0.min())
        Emax = float(Ek_up0.max())

    if energy_window is not None:
        Emin, Emax = energy_window
    else:
        margin = 0.02 * (Emax - Emin)
        Emin, Emax = Emin - margin, Emax + margin

    E_edges   = np.linspace(Emin, Emax, bins + 1)
    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    dE        = E_edges[1] - E_edges[0]

    def _hist_pdf(E):
        # density=True → ∫g(E)dE ≈ 1 for the sample; we re-normalize below.
        g, _ = np.histogram(E, bins=E_edges, density=True)
        return g.astype(float)

    # -------------------- accumulate DOS over shifted meshes ---------------
    if spin_resolved:
        g_up_acc = np.zeros_like(E_centers)
        g_dn_acc = np.zeros_like(E_centers)
    else:
        g_tot_acc = np.zeros_like(E_centers)

    for m in range(n_k_shifts):
        if m == 0:
            sx = sy = 0.0           # include the unshifted mesh
        else:
            sx, sy = rng.random(), rng.random()   # random shift in [0,1)
        KX, KY = _make_mesh(sx, sy)
        Ek_up, Ek_dn = _collect_energies(KX, KY)
        if spin_resolved:
            g_up_acc += _hist_pdf(Ek_up)
            g_dn_acc += _hist_pdf(Ek_dn)
        else:
            g_tot_acc += _hist_pdf(Ek_up)

    if spin_resolved:
        g_up = g_up_acc / n_k_shifts
        g_dn = g_dn_acc / n_k_shifts
        g_tot = g_up + g_dn
    else:
        g_up = g_dn = None
        g_tot = g_tot_acc / n_k_shifts

    # -------------------- optional broadening (choose one) ------------------
    def _apply_kernel(y, k):
        return np.convolve(y, k, mode="same")

    def _gaussian_kernel(sig):
        half = int(np.ceil(4.0 * sig / dE))
        x = np.arange(-half, half + 1) * dE
        k = np.exp(-0.5*(x/sig)**2)
        # normalize area to 1
        k /= (k.sum() * dE)
        return k

    def _lorentzian_kernel(eta):
        half = int(np.ceil(20.0 * eta / dE))  # longer tail support
        x = np.arange(-half, half + 1) * dE
        k = (eta/np.pi) / (x**2 + eta**2)
        k /= (k.sum() * dE)
        return k

    if gaussian_sigma and gaussian_sigma > 0:
        K = _gaussian_kernel(gaussian_sigma)
        if spin_resolved:
            g_up = _apply_kernel(g_up, K)
            g_dn = _apply_kernel(g_dn, K)
            g_tot = g_up + g_dn
        else:
            g_tot = _apply_kernel(g_tot, K)

    if lorentzian_eta and lorentzian_eta > 0:
        K = _lorentzian_kernel(lorentzian_eta)
        if spin_resolved:
            g_up = _apply_kernel(g_up, K)
            g_dn = _apply_kernel(g_dn, K)
            g_tot = g_up + g_dn
        else:
            g_tot = _apply_kernel(g_tot, K)

    # -------------------- normalization (per spin) -------------------------
    if normalize:
        if spin_resolved:
            area_up = max(g_up.sum() * dE, 1e-18)
            area_dn = max(g_dn.sum() * dE, 1e-18)
            g_up /= area_up
            g_dn /= area_dn
            g_tot = g_up + g_dn
        else:
            area = max(g_tot.sum() * dE, 1e-18)
            g_tot /= area

    # -------------------- plotting ----------------------------------------
    # Derive grid size for annotation (from the reference mesh)
    kgrid_nx, kgrid_ny = KX0.shape
    kgrid_text = f"N = {kgrid_nx}×{kgrid_ny}"

    def _finalize_and_save(ax_local, ydata, color, label_text, suffix):
        ax_local.plot(E_centers, ydata, lw=2, label=label_text, color=color)
        ax_local.axvline(mu, ls="--", lw=2, color="green", alpha=0.9, label=r"$\mu$")
        if fill_occupied:
            ax_local.fill_between(E_centers, 0.0, ydata, where=(E_centers <= mu),
                                  color="gray", alpha=0.20, linewidth=0)
        ax_local.set_xlabel(r"energy, $\epsilon/|t|$",fontsize=20)
        ax_local.set_ylabel(r"density of states, $\rho$",fontsize=20)
        ax_local.grid(True, alpha=0.3)
        ax_local.legend(frameon=False,fontsize=22)
        ax.set_xticklabels(fontsize=20)
        ax.set_yticklabels(fontsize=20)
        if annotate_kgrid:
            ax_local.text(0.02, 0.95, kgrid_text, transform=ax_local.transAxes,
                          fontsize=18, va='top', ha='left')
        try:
            setup_plot_formatting(ax_local)
        except Exception:
            pass
        if output_path is not None:
            base, ext = os.path.splitext(output_path)
            outfile = f"{base}{suffix}{ext if ext else '.png'}"
            plt.tight_layout()
            plt.savefig(outfile, dpi=300, bbox_inches='tight')
            plt.close(ax_local.figure)
        return ax_local

    if separate_plots:
        axes_out = {}
        if spin_resolved:
            fig_u, ax_u = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
            axes_out['up'] = _finalize_and_save(ax_u, g_up, colors[0], label_prefix + "$\\uparrow$", "_up")
            fig_d, ax_d = plt.subplots(figsize=(10, 6))
            axes_out['down'] = _finalize_and_save(ax_d, g_dn, colors[1], label_prefix + "$\\downarrow$", "_down")
            fig_t, ax_t = plt.subplots(figsize=(10, 6))
            axes_out['total'] = _finalize_and_save(ax_t, g_tot, "k", label_prefix, "_tot")
            return E_centers, g_up, g_dn, g_tot, axes_out
        else:
            fig_t, ax_t = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
            axes_out = _finalize_and_save(ax_t, g_tot, "k", label_prefix, "")
            return E_centers, None, None, g_tot, axes_out
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        if spin_resolved:
            ax.plot(E_centers, g_up,  lw=2, label=label_prefix + "$\\uparrow$", color=colors[0])
            ax.plot(E_centers, g_dn,  lw=2, label=label_prefix + "$\\downarrow$", color=colors[1])
        else:
            ax.plot(E_centers, g_tot, lw=2, label=label_prefix , color="k")
        ax.axvline(mu, ls="--", lw=2, color="green", alpha=0.9, label=r"$\mu$")
        if fill_occupied and not spin_resolved:
            ax.fill_between(E_centers, 0.0, g_tot, where=(E_centers <= mu),
                            color="gray", alpha=0.20, linewidth=0)
        ax.set_xlabel(r"energy, $\epsilon/|t|$",fontsize=20)
        ax.set_ylabel(r"density of states, $\rho$",fontsize=20)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False,fontsize=22,loc='upper center')
        if annotate_kgrid:
            ax.text(0.396, 0.64, kgrid_text, transform=ax.transAxes,
                    fontsize=22, va='top', ha='left')
        try:
            setup_plot_formatting(ax)
        except Exception:
            pass
        if output_path is not None:
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(ax.figure)
        return E_centers, g_up, g_dn, g_tot, ax


def create_k_path(path_type="gamma_m_gamma", Nk_path=100):
    """
    Create different types of k-paths for 2D square lattice.
    
    Args:
        path_type: Type of path ("gamma_m_gamma", "gamma_x_m_gamma", "gamma_m", "gamma_x")
        Nk_path: Number of points along the path
    
    Returns:
        kx_path, ky_path, tick_positions, tick_labels
    """
    if path_type == "gamma_m_gamma":
        # Γ → M → Γ path: (0,0) → (π,π) → (0,0)
        kx_first = np.linspace(0, np.pi, Nk_path//2)
        ky_first = np.linspace(0, np.pi, Nk_path//2)
        kx_second = np.linspace(np.pi, 0, Nk_path//2)
        ky_second = np.linspace(np.pi, 0, Nk_path//2)
        kx_path = np.concatenate([kx_first, kx_second])
        ky_path = np.concatenate([ky_first, ky_second])
        tick_positions = [0, Nk_path//2, Nk_path-1]
        tick_labels = ['Γ', 'M', 'Γ']

    elif path_type == "gamma_x_m_gamma":
        # Γ → X → M → Γ path: (0,0) → (π,0) → (π,π) → (0,0)
        segment_length = Nk_path // 3
        kx_gx = np.linspace(0, np.pi, segment_length)
        ky_gx = np.zeros(segment_length)
        kx_xm = np.full(segment_length, np.pi)
        ky_xm = np.linspace(0, np.pi, segment_length)
        kx_mg = np.linspace(np.pi, 0, segment_length)
        ky_mg = np.linspace(np.pi, 0, segment_length)
        kx_path = np.concatenate([kx_gx, kx_xm, kx_mg])
        ky_path = np.concatenate([ky_gx, ky_xm, ky_mg])
        tick_positions = [0, segment_length, 2*segment_length, Nk_path-1]
        tick_labels = ['Γ', 'X', 'M', 'Γ']

    elif path_type == "gamma_m":
        # One-way Γ (0,0) → M (π,π)
        kx_path = np.linspace(0, np.pi, Nk_path)
        ky_path = np.linspace(0, np.pi, Nk_path)
        tick_positions = [0, Nk_path-1]
        tick_labels = ['Γ', 'M']

    elif path_type == "gamma_x":
        # One-way Γ (0,0) → X (π,0)
        kx_path = np.linspace(0, np.pi, Nk_path)
        ky_path = np.zeros(Nk_path)
        tick_positions = [0, Nk_path-1]
        tick_labels = ['Γ', 'X']

    else:  # Default to gamma_m_gamma
        kx_path, ky_path, tick_positions, tick_labels = create_k_path("gamma_m_gamma", Nk_path)
    
    return kx_path, ky_path, tick_positions, tick_labels


def plot_occupied_dos(
    dist,
    T,
    h,
    mu,
    *,
    L=120,
    t=-1.0,
    tp=0.25,
    a=1.0,
    n_target=None,
    energy_window=None,
    bins=1200,
    smooth="lorentzian",         # "lorentzian", "gaussian", or None
    gamma=None,                   # for lorentzian (in |t| units). If None, uses 2*dE
    gaussian_sigma=None,          # for gaussian (in |t| units)
    normalize_per_spin=True,
    spin_resolved=True,
    fill_occupied=False,
    label_prefix="",
    ax=None,
    colors=("tab:blue", "tab:orange"),
    output_path=None,
    annotate_kgrid=True,
    separate_plots=True,
):
    """
    Occupied DOS: g_occ(E) = sum_{k,σ} n_{kσ} δ(E - ε_{kσ}).
    Uses μ provided (from dist_data). Not vectorized; loops over k for n_{kσ}.
    Returns E_centers, g_up, g_dn, g_tot, ax_or_axes.
    """
    # k-grid
    kx = np.linspace(0, 2*np.pi/a, L, endpoint=False)
    ky = kx.copy()
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kgrid_text = f"N = {KX.shape[0]}×{KX.shape[1]}"

    # band energies and spin split
    Ek = tight_binding_dispersion(KX, KY, t, tp, a).ravel()
    E_up = Ek - 0.5*h
    E_dn = Ek + 0.5*h

    # occupations via existing nk_* solvers (per-k)
    w_up = np.zeros_like(Ek, dtype=float)
    w_dn = np.zeros_like(Ek, dtype=float)
    flat_KX = KX.ravel()
    flat_KY = KY.ravel()
    for i in range(Ek.size):
        kxi = float(flat_KX[i])
        kyi = float(flat_KY[i])
        if dist == "FD":
            sol = nk_FD(kxi, kyi, T, h, mu, t, tp, a, n_target)
        elif dist == "FLB":
            sol = nk_FLB(kxi, kyi, T, h, mu, t, tp, a, n_target)
        elif dist == "NFL":
            sol = nk_NFL(kxi, kyi, T, h, mu, t, tp, a, n_target)
        else:
            sol = [0.0, 0.0]
        w_up[i] = float(sol[0])
        w_dn[i] = float(sol[1])

    # energy window
    Emin = float(min(E_up.min(), E_dn.min()))
    Emax = float(max(E_up.max(), E_dn.max()))
    if energy_window is not None:
        Emin, Emax = energy_window
    else:
        margin = 0.02 * (Emax - Emin)
        Emin -= margin
        Emax += margin

    E_edges   = np.linspace(Emin, Emax, int(bins) + 1)
    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    dE        = E_edges[1] - E_edges[0]

    # histogram with weights = occupations (counts per bin)
    g_up = np.histogram(E_up, bins=E_edges, weights=w_up)[0].astype(float)
    g_dn = np.histogram(E_dn, bins=E_edges, weights=w_dn)[0].astype(float)
    g_tot = g_up + g_dn

    # smoothing kernels (short support)
    def _gaussian_kernel(dx, sigma, half_std=4.0):
        half = int(np.ceil(half_std * sigma / dx))
        x = np.arange(-half, half + 1) * dx
        k = np.exp(-0.5 * (x / sigma) ** 2)
        k /= max(k.sum(), 1e-18)  # sum=1 to preserve total counts on convolution
        return k

    def _lorentzian_kernel(dx, eta, tail=10.0):
        half = int(np.ceil(tail * eta / dx))
        x = np.arange(-half, half + 1) * dx
        k = (eta / np.pi) / (x**2 + eta**2)
        k /= max(k.sum(), 1e-18)
        return k

    if smooth == "lorentzian":
        if gamma is None:
            gamma = 2.0 * dE
        ker = _lorentzian_kernel(dE, gamma)
        g_up = np.convolve(g_up, ker, mode="same")
        g_dn = np.convolve(g_dn, ker, mode="same")
        g_tot = g_up + g_dn
    elif smooth == "gaussian" and gaussian_sigma is not None and gaussian_sigma > 0:
        ker = _gaussian_kernel(dE, gaussian_sigma)
        g_up = np.convolve(g_up, ker, mode="same")
        g_dn = np.convolve(g_dn, ker, mode="same")
        g_tot = g_up + g_dn

    # normalization per spin to mean occupancy (optional)
    if normalize_per_spin:
        target_up = float(w_up.mean())
        target_dn = float(w_dn.mean())
        area_up = max(g_up.sum() * dE, 1e-18)
        area_dn = max(g_dn.sum() * dE, 1e-18)
        g_up *= (target_up / area_up)
        g_dn *= (target_dn / area_dn)
        g_tot = g_up + g_dn

    # plotting helpers
    def _finalize(ax_local, ydata, color, label_text, suffix):
        ax_local.plot(E_centers, ydata, lw=2, label=label_text, color=color)
        ax_local.axvline(mu, ls="--", lw=2, color="green", alpha=0.9, label=r"$\mu$")
        if fill_occupied and not spin_resolved:
            ax_local.fill_between(E_centers, 0.0, ydata, where=(E_centers <= mu),
                                  color="gray", alpha=0.18, linewidth=0)
        ax_local.set_xlabel(r"energy, $\epsilon/|t|$")
        ax_local.set_ylabel(fr"occupation $\times$ density, $n_{{\bf k}} \times \rho$")
        ax_local.grid(True, alpha=0.3)
        ax_local.legend(frameon=False,loc='upper center',fontsize = 22)
        if annotate_kgrid:
            ax_local.text(0.400, 0.74, kgrid_text, transform=ax_local.transAxes,
                          fontsize=20, va='top', ha='left')
        try:
            setup_plot_formatting(ax_local)
        except Exception:
            pass
        if output_path is not None:
            base, ext = os.path.splitext(output_path)
            out = f"{base}{suffix}{ext if ext else '.png'}"
            plt.tight_layout()
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close(ax_local.figure)
        return ax_local

    if separate_plots and spin_resolved:
        axes_out = {}
        fig_u, ax_u = plt.subplots(figsize=(10, 6)) if ax is None else (ax.figure, ax)
        axes_out['up'] = _finalize(ax_u, g_up, colors[0], label_prefix + "$\\uparrow$", "_occ_up")
        fig_d, ax_d = plt.subplots(figsize=(10, 6))
        axes_out['down'] = _finalize(ax_d, g_dn, colors[1], label_prefix + "$\\downarrow$", "_occ_down")
        return E_centers, g_up, g_dn, g_tot, axes_out

    # combined plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    if spin_resolved:
        ax.plot(E_centers, g_up,  lw=2, label=label_prefix + "$\\uparrow$", color=colors[0])
        ax.plot(E_centers, g_dn,  lw=2, label=label_prefix + "$\\downarrow$", color=colors[1])
    else:
        ax.plot(E_centers, g_tot, lw=2, label=label_prefix, color="k")
    ax.axvline(mu, ls="--", lw=2, color="green", alpha=0.9, label=r"$\mu$")
    if fill_occupied and not spin_resolved:
        ax.fill_between(E_centers, 0.0, g_tot, where=(E_centers <= mu),
                        color="gray", alpha=0.18, linewidth=0)
    ax.set_xlabel(r"energy, $\epsilon/|t|$", fontsize=20)
    ax.set_ylabel(fr"occupation $\times$ density of states, $n_{{\bf k}} \times \rho$",fontsize=20)
    ax.set_xticklabels(fontsize=20)
    ax.set_yticklabels(fontsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False,fontsize=22)
    if annotate_kgrid:
        ax.text(0.02, 0.95, kgrid_text, transform=ax.transAxes,
                fontsize=22, va='top', ha='left')
    try:
        setup_plot_formatting(ax)
    except Exception:
        pass
    if output_path is not None:
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(ax.figure)
    return E_centers, g_up, g_dn, g_tot, ax

def create_high_symmetry_path_plots(distribution_data, output_dir):
    """Create distribution plots along high-symmetry paths (Γ → M → Γ) for each T, n, h combination."""
    
    # Create distribution analysis directory
    distribution_dir = os.path.join(output_dir, "distribution_analysis")
    os.makedirs(distribution_dir, exist_ok=True)
    
    # Group data by T, n_target, h combinations
    from itertools import groupby
    from operator import itemgetter
    
    # Sort by T, n_target, h for grouping
    distribution_data.sort(key=lambda x: (x['T'], x['$n_{target}$'], x['h']))
    
    # Group by T, n_target, h
    grouped_data = {}
    for key, group in groupby(distribution_data, key=lambda x: (x['T'], x['$n_{target}$'], x['h'])):
        T, n_target, h = key
        grouped_data[key] = list(group)
    
    print(f"  Found {len(grouped_data)} unique T, n, h combinations for distribution plotting")
    
    # Process each group
    for (T, n_target, h), group_data in grouped_data.items():
        print(f"  Processing T={T:.6f}, n={n_target:.2f}, h={h:.8f}")
        
        # Create subdirectory for this combination
        # Use very high precision for h to avoid filename collisions/overwrites
        h_precision_dir = 15
        case_dir = os.path.join(distribution_dir, f"T{T:.4f}_n{n_target:.2f}_h{h:.{h_precision_dir}f}")
        os.makedirs(case_dir, exist_ok=True)
        
        # Generate and plot both one-way paths: Γ→M and Γ→X
        for path_choice in ["gamma_m", "gamma_x"]:

            kx_path, ky_path, tick_positions, tick_labels = create_k_path(path_choice, Nk_path)

            # Calculate energies along the path
            energies_path = tight_binding_dispersion(kx_path, ky_path, -1.0, 0.25, 1.0)

            # Calculate occupations for each distribution along this path
            occupations_path = {}
            for dist_data in group_data:
                dist = dist_data['f']
                mu = dist_data['mu']

                n_up_path = []
                n_down_path = []
                n_total_path = []

                for i, (kx, ky) in enumerate(zip(kx_path, ky_path)):
                    # Use self-consistent per-k solvers for spin-resolved occupations
                    if dist == "FD":
                        sol = nk_FD(kx, ky, T, h, mu, -1.0, 0.25, 1.0, n_target)
                    elif dist == "FLB":
                        sol = nk_FLB(kx, ky, T, h, mu, -1.0, 0.25, 1.0, n_target)
                    elif dist == "NFL":
                        sol = nk_NFL(kx, ky, T, h, mu, -1.0, 0.25, 1.0, n_target)
                    else:
                        sol = [0.0, 0.0]

                    n_up = float(sol[0])
                    n_down = float(sol[1])

                    n_up_path.append(n_up)
                    n_down_path.append(n_down)
                    n_total_path.append(n_up + n_down)

                occupations_path[dist] = {
                    'n_up': np.array(n_up_path),
                    'n_down': np.array(n_down_path),
                    'n_total': np.array(n_total_path),
                    'mu': mu
                }

            # Subdirectory per path
            path_dir = os.path.join(case_dir, f"path_{path_choice}")
            os.makedirs(path_dir, exist_ok=True)

            # Create plots for this path with path-specific x-label
            create_distribution_path_plots(
                kx_path, ky_path, energies_path, occupations_path,
                path_dir, T, h, n_target, distributions,
                tick_positions, tick_labels, path_choice
            )

        # Additionally: DOS plots per distribution using the μ saved in group_data
        try:
            dos_dir = os.path.join(case_dir, "DOS")
            os.makedirs(dos_dir, exist_ok=True)
            for dist_data in group_data:
                dist = dist_data['f']
                mu_val = dist_data['mu']
                
                kx_arr = np.linspace(0, 2 * np.pi / a, 2000, endpoint=False)
                ky_arr = kx_arr.copy()
                #kx_arr = dist_data.get('kx')
                #ky_arr = dist_data.get('ky')
                output_path = os.path.join(dos_dir, f"dos_{dist}.png")
                # Use same model parameters as elsewhere in this section
                plot_dos_with_mu(
                    mu=mu_val,
                    kx=kx_arr,
                    ky=ky_arr,
                    t=-1.0,
                    tp=0.25,
                    a=1.0,
                    h=h,
                    spin_resolved=True,
                    bins=800,
                    gaussian_sigma=0.01,
                    fill_occupied=True,
                    output_path=output_path,
                    label_prefix=f"{labels.get(dist, dist)} "
                )

                # Occupied DOS using the same μ
                try:
                    occ_output_base = os.path.join(dos_dir, f"occ_dos_{dist}.png")
                    plot_occupied_dos(
                        dist=dist,
                        T=T,
                        h=h,
                        mu=mu_val,
                        L=N_L,
                        t=-1.0,
                        tp=0.25,
                        a=1.0,
                        n_target=n_target,
                        bins=1200,
                        smooth="lorentzian",
                        gamma=None,
                        gaussian_sigma=None,
                        normalize_per_spin=True,
                        spin_resolved=True,
                        fill_occupied=False,
                        label_prefix=f"{labels.get(dist, dist)} ",
                        separate_plots=True,
                        annotate_kgrid=True,
                        output_path=occ_output_base,
                    )
                except Exception as _e_occ:
                    print(f"      Warning: Occupied DOS failed for dist={dist}: {_e_occ}")
        except Exception as _e:
            print(f"    Warning: DOS plotting failed for T={T:.4f}, n={n_target:.2f}, h={h:.6g}: {_e}")
    
    print(f"  Distribution plots saved in: {distribution_dir}")

def create_distribution_path_plots(kx_path, ky_path, energies_path, occupations_path, 
                                 output_dir, T, h, n_target, distributions, 
                                 tick_positions, tick_labels, path_choice=None):
    """Create various distribution plots along the high-symmetry path."""
    
    # 1. Occupation vs k-path (band structure style)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Calculate k-path distance for x-axis
    # For Γ → M → Γ path, we want a continuous distance measure
    k_distance = np.zeros(len(kx_path))
    
    # Calculate cumulative distance along the path
    for i in range(1, len(kx_path)):
        dkx = kx_path[i] - kx_path[i-1]
        dky = ky_path[i] - ky_path[i-1]
        k_distance[i] = k_distance[i-1] + np.sqrt(dkx**2 + dky**2)
    # Normalize arc length so the one-way path spans [0, π]
    if len(k_distance) > 1 and k_distance[-1] != 0:
        k_scaled = (k_distance / k_distance[-1]) * np.pi
    else:
        k_scaled = k_distance.copy()
    
    # Plot total occupation
    for dist in distributions:
        if dist in occupations_path:
            ax1.plot(k_scaled, occupations_path[dist]['n_total'],
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=3, marker='o', markersize=4,
                    markeredgewidth=1, markeredgecolor='white',
                    label=f'{labels[dist]} (μ={occupations_path[dist]["mu"]:.3f})')
    
    ax1.set_ylabel('total occupation $n_{{\\bf k}\\uparrow} + n_{{\\bf k}\\downarrow}$', fontsize=20)
    ax1.legend(fontsize=18, frameon=False, loc='upper right')
    ax1.grid(True, alpha=0.3)
    setup_plot_formatting(ax1)
    
    # Plot energy bands (overlay spin-split bands ±h/2)
    ax2.plot(k_scaled, energies_path, 'k-', linewidth=2, label='Energy bands (bare)')
    energies_up = energies_path - h/2.0
    energies_down = energies_path + h/2.0
    ax2.plot(k_scaled, energies_up, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='Spin-up band (±h/2)')
    ax2.plot(k_scaled, energies_down, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, label='Spin-down band (±h/2)')
    # Path-specific x label
    if path_choice == "gamma_x":
        x_label_text = r'$\Gamma - X$'
    elif path_choice == "gamma_m":
        x_label_text = r'$\Gamma - M$'
    elif path_choice == "gamma_m_gamma":
        x_label_text = r'$\Gamma - M - \Gamma$'
    else:
        x_label_text = 'k-path distance $|k|$'
    x_label_text = rf'wave vector, ${{\bf k}}$'
    ax2.set_xlabel(x_label_text, fontsize=20)
    ax2.set_ylabel('energy, $\epsilon_k/|t|$', fontsize=20)
    ax2.legend(fontsize=18, frameon=False, loc='upper right')
    ax2.grid(True, alpha=0.3)
    setup_plot_formatting(ax2)
    
    # Set k-path tick labels on the bottom subplot (shared x-axis)
    ax2.set_xticks([k_scaled[i] for i in tick_positions])
    ax2.set_xticklabels(tick_labels, fontsize=20)
    ax2.set_xlim(0.0, np.pi)
    
    # Add parameter annotations
    temp_text = format_temperature_annotation(T)
    h_text = rf"$\frac{{\mu_B H}}{{|t|}} = {format_magnetic_field(h)}$"
    n_text = f"$n = {n_target:.2f}$"
    annotation_text = f"{temp_text}\n{h_text}\n{n_text}"
    
    ax1.text(0.54, 0.95, annotation_text, transform=ax1.transAxes,
            fontsize=18, verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    
    # Use higher precision for very small temperatures and h values
    if T < 0.01:
        T_precision = 4
    else:
        T_precision = 3
    
    if h < 0.001:
        h_precision = 8
    elif h < 0.01:
        h_precision = 6
    else:
        h_precision = 4
    
    filename = os.path.join(output_dir, f"occupation_vs_kpath_T{T:.{T_precision}f}_n{n_target:.2f}_h{h:.{h_precision}f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spin-resolved occupations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    for dist in distributions:
        if dist in occupations_path:
            ax1.plot(k_scaled, occupations_path[dist]['n_up'],
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=3, marker='o', markersize=4,
                    markeredgewidth=1, markeredgecolor='white',
                    label=f'{labels[dist]} ↑')
            
            ax2.plot(k_scaled, occupations_path[dist]['n_down'],
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=3, marker='o', markersize=4,
                    markeredgewidth=1, markeredgecolor='white',
                    label=f'{labels[dist]} ↓')
    
    ax1.set_ylabel('occupation number, $n_{{\\bf k}\\uparrow}$', fontsize=20)
    ax1.legend(fontsize=18, frameon=False, loc='upper right')
    ax1.grid(True, alpha=0.3)
    setup_plot_formatting(ax1)
    
    ax2.set_xlabel(x_label_text, fontsize=20)
    ax2.set_ylabel('occupation number, $n_{{\\bf k}\\downarrow}$', fontsize=20)
    ax2.legend(fontsize=18, frameon=False, loc='upper right')
    ax2.grid(True, alpha=0.3)
    setup_plot_formatting(ax2)
    
    # Set k-path tick labels on the bottom subplot (shared x-axis)
    ax2.set_xticks([k_scaled[i] for i in tick_positions])
    ax2.set_xticklabels(tick_labels, fontsize=20)
    #ax2.set_yticklabels(fontsize=20)
    #ax1.set_yticklabels(fontsize=20)

    ax2.set_xlim(0.0, np.pi)
    
    # Add parameter annotations
    ax1.text(0.64, 0.95, annotation_text, transform=ax1.transAxes,
            fontsize=18, verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"spin_resolved_vs_kpath_T{T:.{T_precision}f}_n{n_target:.2f}_h{h:.{h_precision}f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Distribution comparison (all on same plot)
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for dist in distributions:
        if dist in occupations_path:
            ax.plot(k_scaled, occupations_path[dist]['n_total'],
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=3, marker='o', markersize=6,
                    markeredgewidth=1.5, markeredgecolor='white',
                    label=f'{labels[dist]} (μ={occupations_path[dist]["mu"]:.3f})')
    
    ax.set_xlabel(x_label_text, fontsize=20)
    ax.set_ylabel('total occupation number, $n_{{\\bf k},\\uparrow} + n_{{\\bf k},\\downarrow}$', fontsize=20)
    ax.legend(fontsize=18, frameon=False, loc='upper right')
    ax.grid(True, alpha=0.3)
    setup_plot_formatting(ax)
    
    # Set k-path tick labels
    ax.set_xticks([k_scaled[i] for i in tick_positions])
    ax.set_xticklabels(tick_labels, fontsize=20)
    #ax.set_yticklabels(fontsize=20)
    ax.set_xlim(0.0, np.pi)
    
    # Add parameter annotations
    ax.text(0.73, 0.79, annotation_text, transform=ax.transAxes,
            fontsize=18, verticalalignment='top', horizontalalignment='left')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"distribution_comparison_T{T:.{T_precision}f}_n{n_target:.2f}_h{h:.{h_precision}f}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Distribution plots saved for T={T:.4f}, n={n_target:.2f}, h={h:.6f}")

# Generate distribution plots using collected data
create_high_symmetry_path_plots(distribution_data, output_dir_thermo)

# 6. ENHANCED DISTRIBUTION PLOTS (both self-consistent and comparison)
print("Generating enhanced distribution plots...")

# Select representative cases for distribution plotting
repr_cases = [
    {'T': 0.01, 'h': 0.0, 'n_target': 0.8},  # Low T, no field, high filling
    {'T': 0.1, 'h': 0.005, 'n_target': 0.5},  # Medium T, small field, half filling
    {'T': 0.001, 'h': 0.0007, 'n_target': 0.99},  # Ultra-low T, critical field, high filling
]

"""
def create_distribution_comparison_plot(data_dict, output_dir, distributions, T, h, mu_fd, n_target):
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    energies = data_dict['energies']
    sort_indices = np.argsort(energies)
    energies_sorted = energies[sort_indices]
    
    for dist in distributions:
        if dist in data_dict['occupations']:
            occupations = data_dict['occupations'][dist]
            n_total_sorted = occupations['n_total'][sort_indices]
            
            ax.plot(energies_sorted, n_total_sorted,
                    color=colors[dist], linestyle=styles[dist],
                    linewidth=4, marker='o', markersize=6,
                    markeredgewidth=1.5, markeredgecolor='white',
                    label=f'{labels[dist]}',
                    alpha=0.9)
    
    # Add vertical line at chemical potential
    ax.axvline(x=mu_fd, color='black', linestyle=':', linewidth=3, alpha=0.8, 
              label=f'$\\mu_{{FD}} = {mu_fd:.3f}$')
    
    ax.set_xlabel('Energy $\\epsilon_k/|t|$', fontsize=18)
    ax.set_ylabel('Total Occupation $n_k^\\uparrow + n_k^\\downarrow$', fontsize=18)
    # Remove title and add annotations
    add_plot_annotations(ax, T, h, n_target, x_pos=0.02, y_pos=0.95)
    
    ax.legend(fontsize=14, frameon=False, loc='upper right')
    setup_plot_formatting(ax)
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"distribution_comparison_fixed_mu.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Custom comparison plot saved: {filename}")
"""

"""
for i, case in enumerate(repr_cases):
    T, h, n_target = case['T'], case['h'], case['n_target']
    case_dir = os.path.join(analysis_dirs["distributions"], f"case_{i+1}_T{T}_h{h}_n{n_target}")
    os.makedirs(case_dir, exist_ok=True)
    
    print(f"  Generating distribution plots for T={T}, h={h}, n={n_target}")
    
    # METHOD 1: Self-consistent plots (each distribution uses its own μ)
    self_consistent_dir = os.path.join(case_dir, "self_consistent")
    os.makedirs(self_consistent_dir, exist_ok=True)
    
    try:
        plot_all_distributions(T, h, n_target, Nk_test, -1.0, 0.25, 1.0, distributions, self_consistent_dir)
        print(f"    Self-consistent plots saved in {self_consistent_dir}")
    except Exception as e:
        print(f"    Warning: Could not generate self-consistent plots: {e}")
    
    # METHOD 2: Comparison plots (all distributions use FD chemical potential)
    comparison_dir = os.path.join(case_dir, "fixed_mu_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # Solve for FD chemical potential
        mu_fd = solve_mu_FD(n_target, T, Nk_test, h, -1.0, 0.25, 1.0)
        print(f"    FD chemical potential: μ_FD = {mu_fd:.6f}")
        
        # Compute distribution data using FD chemical potential for all distributions
        data_comparison = compute_distribution_data(T, h, mu_fd, Nk_test, -1.0, 0.25, 1.0, distributions)
        
        # Create comparison plots
        plot_distribution_vs_energy(data_comparison, comparison_dir, distributions, T, h, mu_fd, -1.0, 0.25, n_target)
        plot_distribution_vs_momentum(data_comparison, comparison_dir, distributions, T, h, mu_fd, -1.0, 0.25, n_target)
        plot_distribution_heatmap(data_comparison, comparison_dir, distributions, T, h, mu_fd, -1.0, 0.25, n_target, Nk_test)
        
        # Create custom comparison plot
        create_distribution_comparison_plot(data_comparison, comparison_dir, distributions, T, h, mu_fd, n_target)
        
        print(f"    Comparison plots saved in {comparison_dir}")
    except Exception as e:
        print(f"    Warning: Could not generate comparison plots: {e}")
"""
print("\n" + "="*60)
print("UNDERSTANDING THE DISTRIBUTION DIFFERENCES")
print("="*60)
print("📌 KEY INSIGHT:")
print("  Two types of distribution plots are generated:")
print()
print("  1. SELF-CONSISTENT PLOTS:")
print("     - Each distribution (FD, FLB, NFL) uses its own μ")
print("     - All achieve the same target density n_target")
print("     - Distributions appear 'shifted' because μ_FD ≠ μ_FLB ≠ μ_NFL")
print("     - This is PHYSICALLY MEANINGFUL - shows how much μ must change")
print("       to achieve the same density with different statistics")
print()
print("  2. COMPARISON PLOTS (Fixed μ_FD):")
print("     - All distributions use the same μ (from Fermi-Dirac)")
print("     - Shows 'pure' effect of different distribution functions")
print("     - NFL typically has HIGHER occupation than FD at same μ")
print("     - FLB typically has INTERMEDIATE occupation")
print("     - These plots show the statistical differences clearly")
print()
print("🔬 PHYSICAL INTERPRETATION:")
print("  - NFL needs LOWER μ to achieve same density (μ_NFL < μ_FD)")
print("  - FLB needs μ between FD and NFL (μ_NFL < μ_FLB < μ_FD)")
print("  - This reflects the enhanced/reduced occupation tendencies")
print("  - The 'shifting' you observed is correct behavior!")

print("\n🔍 DIAGNOSTIC: Why comparison plots show no differences")
print("="*60)

# Test parameters
#T = 0.01
#h = 0.0
#n_target = 0.8
#Nk = 400
#t = -1.0
#tp = 0.25
#a = 1.0

print(f"Test conditions: T={T}, h={h}, n_target={n_target}, Nk={Nk}")

# Step 1: Find individual chemical potentials (self-consistent)
print("\n1️⃣ STEP 1: Individual chemical potentials")
mu_dict = {}
for dist in distributions:
    if dist == "FD":
        mu = solve_mu_FD(n_target, T, Nk, h, t, tp, a)
    elif dist == "FLB":
        mu = solve_mu_FLB(n_target, T, Nk, h, t, tp, a)
    elif dist == "NFL":
        mu = solve_mu_NFL(n_target, T, Nk, h, t, tp, a)
    mu_dict[dist] = mu
    print(f"  {dist}: μ = {mu:.6f}")

print(f"\nμ differences: μ_FD - μ_FLB = {mu_dict['FD'] - mu_dict['FLB']:.6f}")
print(f"                μ_FD - μ_NFL = {mu_dict['FD'] - mu_dict['NFL']:.6f}")

# Step 2: Use FD chemical potential for all distributions
print("\n2️⃣ STEP 2: Fixed μ_FD comparison")
mu_fd = mu_dict['FD']
print(f"Using μ_FD = {mu_fd:.6f} for all distributions")

# Step 3: Sample a few k-points manually to debug
print("\n3️⃣ STEP 3: Manual k-point sampling")
k_sample_indices = [(0, 0), (1, 1), (2, 2)]  # Sample a few k-points

for i, j in k_sample_indices:
    kx = i * 2 * np.pi / Nk
    ky = j * 2 * np.pi / Nk
    energy = tight_binding_dispersion(kx, ky, t, tp, a)
    
    print(f"\nk-point ({i},{j}): kx={kx:.3f}, ky={ky:.3f}, ε={energy:.3f}")
    
    for dist in distributions:
        if dist == "FD":
            sol = nk_FD(kx, ky, T, h, mu_fd, t, tp, a, n_target)
        elif dist == "FLB":
            sol = nk_FLB(kx, ky, T, h, mu_fd, t, tp, a, n_target)
        elif dist == "NFL":
            sol = nk_NFL(kx, ky, T, h, mu_fd, t, tp, a, n_target)
        
        n_total = sol[0] + sol[1]
        print(f"  {dist}: n_up={sol[0]:.4f}, n_down={sol[1]:.4f}, n_total={n_total:.4f}")

# Step 4: Generate full comparison plot with enhanced diagnostics
print("\n4️⃣ STEP 4: Full comparison plot")

# Generate data using the existing function
data_comparison = compute_distribution_data(T, h, mu_fd, Nk, t, tp, a, distributions, n_target)

energies = data_comparison['energies']
sort_indices = np.argsort(energies)
energies_sorted = energies[sort_indices]

print(f"Energy range: {energies.min():.3f} to {energies.max():.3f}")
print(f"Chemical potential μ_FD = {mu_fd:.3f}")
print(f"μ_FD position: {'INSIDE' if energies.min() < mu_fd < energies.max() else 'OUTSIDE'} energy range")
"""
# Calculate some statistics
total_densities = {}
for dist in distributions:
    n_total = data_comparison['occupations'][dist]['n_total']
    total_densities[dist] = np.mean(n_total)
    print(f"Average density for {dist}: {total_densities[dist]:.4f}")

# Create enhanced diagnostic plot with improved formatting
fig = plt.figure(figsize=(20, 16))

# Main plot: distributions vs energy
ax1 = plt.subplot(2, 2, 1)
for dist in distributions:
    n_total_sorted = data_comparison['occupations'][dist]['n_total'][sort_indices]
    ax1.plot(energies_sorted, n_total_sorted,
            color=colors[dist], linestyle=styles[dist],
            linewidth=4, marker='o', markersize=6,
            markeredgewidth=1.5, markeredgecolor='white',
            label=f'{labels[dist]} ($\\mu={mu_dict[dist]:.3f}$)')

ax1.axvline(x=mu_fd, color='black', linestyle=':', linewidth=3, alpha=0.8, 
           label=f'$\\mu_{{FD}} = {mu_fd:.3f}$')
ax1.set_xlabel('Energy $\\epsilon_k/|t|$', fontsize=18)
ax1.set_ylabel('Total Occupation $n_k$', fontsize=18)
ax1.set_title('Comparison Plot: All use $\\mu_{FD}$', fontsize=20)
ax1.legend(fontsize=12, frameon=False, loc='upper right')
setup_plot_formatting(ax1)

# Diagnostic plot 1: Difference from FD
ax2 = plt.subplot(2, 2, 2)
n_fd_sorted = data_comparison['occupations']['FD']['n_total'][sort_indices]
for dist in distributions:
    if dist != 'FD':
        n_dist_sorted = data_comparison['occupations'][dist]['n_total'][sort_indices]
        diff = n_dist_sorted - n_fd_sorted
        ax2.plot(energies_sorted, diff,
                color=colors[dist], linestyle=styles[dist],
                linewidth=3, marker='o', markersize=5,
                markeredgewidth=1, markeredgecolor='white',
                label=f'{labels[dist]} - FD')

ax2.axhline(y=0, color='black', linestyle='-', alpha=0.6, linewidth=2)
ax2.axvline(x=mu_fd, color='black', linestyle=':', alpha=0.8, linewidth=2)
ax2.set_xlabel('Energy $\\epsilon_k/|t|$', fontsize=18)
ax2.set_ylabel('Difference from FD', fontsize=18)
ax2.set_title('Occupation Differences', fontsize=20)
ax2.legend(fontsize=12, frameon=False, loc='upper right')
setup_plot_formatting(ax2)

# Diagnostic plot 2: Energy histogram
ax3 = plt.subplot(2, 2, 3)
ax3.hist(energies, bins=20, alpha=0.7, color='gray', edgecolor='black', linewidth=1.5)
ax3.axvline(x=mu_fd, color='red', linestyle='--', linewidth=3, 
           label=f'$\\mu_{{FD}} = {mu_fd:.3f}$')
ax3.set_xlabel('Energy $\\epsilon_k/|t|$', fontsize=18)
ax3.set_ylabel('Count', fontsize=18)
ax3.set_title('Energy Distribution', fontsize=20)
ax3.legend(fontsize=12, frameon=False, loc='upper right')
setup_plot_formatting(ax3)

# Diagnostic plot 3: Occupations near μ
ax4 = plt.subplot(2, 2, 4)
# Find points near chemical potential
near_mu_mask = np.abs(energies - mu_fd) < 1.0  # Within 1.0 of μ
"""
"""
if np.any(near_mu_mask):
    energies_near = energies[near_mu_mask]
    sort_near = np.argsort(energies_near)
    energies_near_sorted = energies_near[sort_near]
    
    for dist in distributions:
        n_near = data_comparison['occupations'][dist]['n_total'][near_mu_mask]
        n_near_sorted = n_near[sort_near]
        ax4.plot(energies_near_sorted, n_near_sorted,
                color=colors[dist], linestyle=styles[dist],
                linewidth=4, marker='o', markersize=8,
                markeredgewidth=1.5, markeredgecolor='white',
                label=f'{labels[dist]}')
    
    ax4.axvline(x=mu_fd, color='black', linestyle=':', linewidth=3, alpha=0.8)
    ax4.set_xlabel('Energy $\\epsilon_k/|t|$ (near $\\mu$)', fontsize=18)
    ax4.set_ylabel('Total Occupation $n_k$', fontsize=18)
    ax4.set_title('Zoom: Near Chemical Potential', fontsize=20)
    ax4.legend(fontsize=12, frameon=False, loc='upper right')
    setup_plot_formatting(ax4)
else:
    ax4.text(0.5, 0.5, 'No points near $\\mu_{FD}$', ha='center', va='center', 
            transform=ax4.transAxes, fontsize=16)

plt.tight_layout()
plt.savefig('distribution_comparison_diagnostic.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n5️⃣ STEP 5: Analytical check")
print("For T=0.01, these distributions should differ significantly:")
"""
# Create analytical comparison at a specific energy
test_energy = mu_fd
e_up = test_energy - h - mu_fd  # = test_energy - h - mu_fd 
e_down = test_energy + h - mu_fd  # = test_energy + h - mu_fd

print(f"At ε = μ_FD = {test_energy:.3f}:")
print(f"  e_up = ε - h - μ = {e_up:.6f}")
print(f"  e_down = ε + h - μ = {e_down:.6f}")

# FD analytical
n_fd_up = 1.0 / (1.0 + np.exp(e_up / T))
n_fd_down = 1.0 / (1.0 + np.exp(e_down / T))
print(f"  FD: n_up = {n_fd_up:.4f}, n_down = {n_fd_down:.4f}, n_total = {n_fd_up + n_fd_down:.4f}")

print("\n🎯 EXPECTED BEHAVIOR:")
print("- At ε = μ, FD should give n ≈ 0.5 per spin")
print("- NFL should give n > FD (enhanced occupation)")
print("- FLB should give intermediate values")
print("- If all distributions give similar values, there's a bug!")

print("\n📊 Analysis complete. Check the diagnostic plot!")

print("\n" + "="*60)
print("QUANTUM LIMIT TEST COMPLETED WITH ENHANCED DISTRIBUTION ANALYSIS!")
print("="*60)
print("🎯 RECOMMENDATION:")
print("  - Use COMPARISON PLOTS to see pure statistical effects")
print("  - Use SELF-CONSISTENT PLOTS to see thermodynamic differences")
print("  - The 'shifting' is not a bug - it's the physics!")

print("\n" + "="*60)
print("MAGNETIZATION ANALYSIS ADDED!")
print("="*60)
print("🔧 NEW FEATURES:")
print("  1. Magnetization calculation added to main thermodynamic loop")
print("     - M = n_up - n_down for each state")
print("     - Stored in thermo_data DataFrame")
print()
print("  2. Magnetization plots generated:")
print("     - M vs h for specific T and n_target combinations")
print("     - Both regular and scaled (×10⁴) h-axis versions")
print("     - Professional formatting with annotations")
print()
print("  3. Broken axis magnetization plots:")
print("     - Uses collected thermodynamic data")
print("     - Creates plots for consecutive temperature pairs: (0.0001, 0.001), (0.001, 0.01), etc.")
print("     - Each pair shows the evolution of magnetization across temperature scales")
print("     - Automatically handles data filtering and formatting")
print("     - Dynamic scaling adapts to each temperature pair's h-field range")
print()
print("  4. Distribution plots along high-symmetry paths:")
print("     - Γ → M → Γ path for each T, n, h combination")
print("     - Uses self-consistent μ values from main loop")
print("     - Band structure style plots with occupation vs k-path")
print("     - Spin-resolved occupations (n↑, n↓ separately)")
print("     - Distribution comparison plots")
print()
print("📁 OUTPUT LOCATION:")
print(f"  Magnetization plots: {output_dir_thermo}/magnetization_analysis/")
print(f"  Distribution plots: {output_dir_thermo}/distribution_analysis/")
print()
print("🎛️ CUSTOMIZATION:")
print("  - Modify 'magnetization_temperatures' and 'magnetization_n_targets' to change T/n_target combinations")
print("  - Broken axis plots automatically created for all consecutive temperature pairs")
print("  - Dynamic scaling automatically adapts to h-field ranges for optimal visibility")
print("  - Distribution plots automatically generated for all T, n, h combinations")
print("  - All parameters are automatically derived from simulation parameters to ensure data availability")
print("  - All plots use consistent formatting and color schemes")
print("  - File names include appropriate precision for T and h values") 

def test_parallel_implementation():
    """
    Test function to verify the parallel implementation works correctly.
    """
    print("\n" + "="*60)
    print("TESTING PARALLEL IMPLEMENTATION")
    print("="*60)
    
    # Test parameters from config
    test_params = CONFIG['testing']['test_parameters']
    test_temperatures = test_params['temperatures']
    test_h_values = test_params['h_values']
    test_n_targets = test_params['n_targets']
    test_distributions = test_params['distributions']
    test_Nk = test_params['Nk']
    
    print(f"Test parameters:")
    print(f"  Temperatures: {test_temperatures}")
    print(f"  H values: {test_h_values}")
    print(f"  N targets: {test_n_targets}")
    print(f"  Distributions: {test_distributions}")
    print(f"  Nk: {test_Nk}")
    
    # Test parallel processing
    print(f"\nTesting PARALLEL processing...")
    start_time = time.time()
    parallel_results = collect_thermo_data_parallel(
        test_temperatures, 
        test_h_values, 
        test_n_targets, 
        test_distributions, 
        test_Nk, 
        -1.0, 0.25, 1.0, 
        2  # Use 2 cores for testing
    )
    parallel_time = time.time() - start_time
    
    # Test sequential processing
    print(f"\nTesting SEQUENTIAL processing...")
    start_time = time.time()
    sequential_results = []
    for T in test_temperatures:
        for h in test_h_values:
            for n_target in test_n_targets:
                for dist in test_distributions:
                    kx = np.linspace(0, 2 * np.pi, test_Nk, endpoint=False)
                    ky = kx.copy()
                    result = one_state(dist, T, n_target, h, kx, ky, -1.0, 0.25, 1.0)
                    mu = result['mu']
                    n_up = result['n_up']
                    n_down = result['n_down']
                    entropy = compute_total_entropy(T, h, mu, test_Nk, -1.0, 0.25, 1.0, dist, n_target)
                    
                    sequential_results.append({
                        'f': dist,
                        'T': T,
                        '$n_{target}$': n_target,
                        'h': h,
                        'mu': mu,
                        'n_up': n_up,
                        'n_down': n_down,
                        'M': n_up - n_down,
                        'S': entropy,
                    })
    sequential_time = time.time() - start_time
    
    # Compare results
    print(f"\nResults comparison:")
    print(f"  Parallel time: {parallel_time:.3f}s")
    print(f"  Sequential time: {sequential_time:.3f}s")
    print(f"  Speedup: {sequential_time/parallel_time:.2f}x")
    print(f"  Parallel results: {len(parallel_results)}")
    print(f"  Sequential results: {len(sequential_results)}")
    
    # Check if results match
    if len(parallel_results) == len(sequential_results):
        print(f"  ✓ Result counts match")
        
        # Compare individual results
        matches = 0
        for p_result, s_result in zip(parallel_results, sequential_results):
            if (abs(p_result['mu'] - s_result['mu']) < 1e-6 and
                abs(p_result['S'] - s_result['S']) < 1e-6 and
                abs(p_result['M'] - s_result['M']) < 1e-6):
                matches += 1
        
        print(f"  ✓ {matches}/{len(parallel_results)} individual results match")
        
        if matches == len(parallel_results):
            print(f"  🎉 PARALLEL IMPLEMENTATION TEST PASSED!")
        else:
            print(f"  ⚠️  Some results don't match - check implementation")
    else:
        print(f"  ❌ Result counts don't match - check implementation")
    
    return parallel_results, sequential_results

# Run test if enabled in config
if CONFIG['testing']['run_test']:
    test_parallel_implementation()

# Uncomment the following block to run diagnostics
# if CONFIG.get('run_diagnostics', False):
#     # All the diagnostic code here
#     pass

# === CRITICAL FIELD ANALYSIS AND PLOT ===

def find_hc_jump(h, M, flank=2, k_sigma=3, min_jump=2e-5):
    """
    Robust h_c from a linear background + jump:
    - estimates baseline slope from dM/dh (median),
    - finds the largest excess slope over baseline,
    - estimates pre/post plateaus from small windows,
    - interpolates between the two samples that straddle the jump
      to the midpoint between plateaus (not n_target/2).
    """
    h = np.asarray(h); M = np.asarray(M)
    idx = np.argsort(h)
    h = h[idx]; M = M[idx]
    # de-duplicate h
    h, uidx = np.unique(h, return_index=True)
    M = M[uidx]
    if len(h) < 4:
        return np.nan, None

    # derivative and robust baseline
    dM = np.gradient(M, h)
    baseline = np.median(dM)
    mad = 1.4826 * np.median(np.abs(dM - baseline)) + 1e-12

    # peak excess over baseline
    excess = dM - baseline
    peak = np.argmax(excess)
    if excess[peak] < k_sigma * mad:
        return np.nan, dM  # no significant jump

    # require meaningful discrete jump between neighbors
    if peak >= len(M) - 1:
        return np.nan, dM
    jump_mag = M[peak+1] - M[peak]
    if jump_mag < min_jump:
        return np.nan, dM

    # pre/post plateaus via small windows
    L0 = max(0, peak - flank); L1 = peak
    R0 = peak + 1; R1 = min(len(M), peak + 1 + flank)
    M_low = np.median(M[L0:L1]) if L1 > L0 else M[peak]
    M_high = np.median(M[R0:R1]) if R1 > R0 else M[peak+1]

    # midpoint between observed plateaus (not tied to n_target)
    target_M = 0.5 * (M_low + M_high)

    # linear interpolation between the two points that straddle the jump
    h0, h1 = h[peak], h[peak+1]
    M0, M1 = M[peak], M[peak+1]
    if M1 == M0:
        hc = 0.5 * (h0 + h1)
    else:
        hc = h0 + (target_M - M0) * (h1 - h0) / (M1 - M0)

    return hc, dM


def find_hc_fallback_midpoint(h, M, min_jump=1e-5):
    """
    Fallback: if a clear jump isn't detected, compute the midpoint between
    global min and max of M and return the first crossing by linear interpolation.
    """
    h = np.asarray(h); M = np.asarray(M)
    idx = np.argsort(h)
    h = h[idx]; M = M[idx]
    if len(h) < 2:
        return np.nan
    Mmin = np.min(M); Mmax = np.max(M)
    if (Mmax - Mmin) < min_jump:
        return np.nan
    target_M = 0.5 * (Mmin + Mmax)
    for i in range(len(M) - 1):
        M0, M1 = M[i], M[i+1]
        if (M0 - target_M) * (M1 - target_M) <= 0 and (M1 != M0):
            return h[i] + (target_M - M0) * (h[i+1] - h[i]) / (M1 - M0)
    return np.nan

# Use existing temperatures_extended and run for both n=0.50 and n=0.95
n_targets_hc = [0.50, 0.95]  # Both n values:w
#n_targets_hc = [0.96, 0.98]  # Both n values:w

hc_vs_T = []
for T in temperatures_extended:
    for n_target in n_targets_hc:
        df_T = df_thermo[(df_thermo['T'] == T) & (df_thermo['$n_{target}$'] == n_target) & (df_thermo['f'] == 'NFL')]
        if len(df_T) < 3:
            print(f"Skipping T={T}, n={n_target}: not enough points for gradient method.")
            continue
        h = df_T['h'].values
        M = df_T['M'].values
        hc, _ = find_hc_jump(h, M)
        if np.isnan(hc):
            hc_fb = find_hc_fallback_midpoint(h, M)
            if not np.isnan(hc_fb):
                hc = hc_fb
                print(f"T={T:.5f}, n={n_target:.2f}, h_c(fallback)={hc:.8f}")
            else:
                print(f"Skipping T={T}, n={n_target}: no detectable jump within h-range [{h.min():.8e}, {h.max():.8e}]. Consider widening TEMPERATURE_SPECIFIC_FIELDS_BY_N for this (n,T).")
                continue
        hc_vs_T.append((T, n_target, hc))
        if not np.isnan(hc):
            print(f"T={T:.5f}, n={n_target:.2f}, h_c={hc:.8f}")

# Plot h_c(T) for both n_targets
if hc_vs_T:
    import itertools
    plt.figure(figsize=(8,6))
    
    # Colors for different n values
    colors = {0.50: 'green', 0.95: 'red'}
    #colors = {0.96: 'green', 0.98: 'red'}

    for n_target, group in itertools.groupby(sorted(hc_vs_T, key=lambda x: x[1]), key=lambda x: x[1]):
        group = list(group)
        T_vals = [x[0] for x in group]
        hc_vals = [x[2] for x in group]
        if len(T_vals) > 0 and len(hc_vals) > 0:
            plt.plot([v * 10 for v in T_vals], [v * 1e4 for v in hc_vals], 'o-', 
                    label=fr'$n={n_target:.2f}$', color=colors[n_target])
            
            # Add trend line for n=0.95 through first 3 points
            if n_target == 0.95 and len(T_vals) >= 3:
                T_trend = [v * 10 for v in T_vals[:3]]
                hc_trend = [v * 1e4 for v in hc_vals[:3]]
                # Fit linear trend
                coeffs = np.polyfit(T_trend, hc_trend, 1)
                T_trend_line = np.linspace(0, max(T_trend)*10, 100)
                hc_trend_line = coeffs[0] * T_trend_line + coeffs[1]
                plt.plot(T_trend_line, hc_trend_line, '--', color='blue', 
                        linewidth=2, label='linear')
    
    plt.xlabel(r'temperature, ${k_B T}/{|t|} \times 10$')
    plt.ylabel(r'critical field, $\mu_B H_c/|t| \times 10^4$')
    plt.xlim(0.0, None)  # Start temperature axis from 0.0
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('hc_vs_T_gradient_method.png', dpi=300)
    plt.show()
    print("Saved h_c(T) plot as 'hc_vs_T_gradient_method.png'")

    # --- NEW: Plot T vs h_c (axes swapped) ---
    plt.figure(figsize=(8,6))
    for n_target, group in itertools.groupby(sorted(hc_vs_T, key=lambda x: x[1]), key=lambda x: x[1]):
        group = list(group)
        T_vals = [x[0] for x in group]
        hc_vals = [x[2] for x in group]
        if len(T_vals) > 0 and len(hc_vals) > 0:
            plt.plot([v * 1e4 for v in hc_vals], [v * 10 for v in T_vals], 'o-', 
                    label=fr'$n={n_target:.2f}$', color=colors[n_target])
            
            # Add trend line for n=0.95 through first 3 points (swapped axes)
            if n_target == 0.95 and len(T_vals) >= 3:
                hc_vals_scaled = np.array(hc_vals) * 1e4  # Scale all hc_vals once
                hc_trend = [v * 1e4 for v in hc_vals[:3]]
                T_trend = [v * 10 for v in T_vals[:3]]
                # Fit linear trend (swapped variables)
                coeffs = np.polyfit(hc_trend, T_trend, 1)
                hc_trend_line = np.linspace(min(hc_vals_scaled), max(hc_vals_scaled), 100)
                T_trend_line = coeffs[0] * hc_trend_line + coeffs[1]
                plt.plot(hc_trend_line, T_trend_line, '--', color='blue', 
                        linewidth=2, label='linear')
    
    plt.xlabel(r'critical field, $\mu_B H_c/|t| \times 10^4$')
    plt.ylabel(r'temperature, ${k_B T}/{|t|} \times 10$')
    plt.ylim(0.0, None)  # Start temperature axis from 0.0
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('T_vs_hc_gradient_method.png', dpi=300)
    plt.show()
    print("Saved T(h_c) plot as 'T_vs_hc_gradient_method.png'")
else:
    print("No h_c(T) data to plot.")

