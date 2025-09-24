import numpy as np
from scipy.optimize import minimize, root_scalar
import math

# Import core objective functions and entropy from the main module
from chartformat_freeenergy_SPIN_ALMOST_Correct_Magnetization_Entropy3 import (
    func_FD, func_deriv_FD, cons_FD,
    func_FLB, func_deriv_FLB, cons_FLB,
    func_NFL, func_deriv_NFL, cons_NFL,
    entropy_k_point,
)


# =============================== FD ===================================== #
def nk_FD_split(kx, ky, T, h, mu, t_up, tp_up, t_dn, tp_dn, a=1.0, n_target=None):
    if n_target is None:
        raise ValueError("nk_FD_split requires n_target for initial guess")

    coskx = np.cos(kx)
    cosky = np.cos(ky)
    # Spin-resolved dispersions (nearest + next-nearest)
    e_up = 2.0 * t_up * (coskx + cosky) + 4.0 * tp_up * coskx * cosky - h/2.0 - mu
    e_down = 2.0 * t_dn * (coskx + cosky) + 4.0 * tp_dn * coskx * cosky + h/2.0 - mu

    initial_guess = [n_target/2.0, n_target/2.0]
    res = minimize(
        func_FD,
        initial_guess,
        args=(e_up, e_down, T),
        jac=func_deriv_FD,
        constraints=cons_FD,
        method='SLSQP',
        options={'eps': 1e-10, 'disp': False}
    )
    if not res.success:
        raise RuntimeError(f"FD minimize failed at k=({kx},{ky}): {res.message}")
    return res.x


def n_tot_FD_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a=1.0, n_target=None):
    sum_up = 0.0
    sum_dn = 0.0
    for i in range(L):
        kx = i * 2.0 * math.pi / L
        for j in range(L):
            ky = j * 2.0 * math.pi / L
            n_up, n_dn = nk_FD_split(kx, ky, T, h, mu, t_up, tp_up, t_dn, tp_dn, a, n_target)
            sum_up += n_up
            sum_dn += n_dn
    return sum_up / (L*L), sum_dn / (L*L)


def density_FD_split(mu, T, h, L, n_target, t_up, tp_up, t_dn, tp_dn, a=1.0):
    n_up, n_dn = n_tot_FD_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, n_target)
    return n_up + n_dn - n_target


def solve_mu_FD_split(n_target, T, L, h, t_up, tp_up, t_dn, tp_dn, a=1.0, mu_low=-80, mu_high=80):
    f = lambda mu: density_FD_split(mu, T, h, L, n_target, t_up, tp_up, t_dn, tp_dn, a)
    return root_scalar(f, bracket=[mu_low, mu_high], method='brentq').root


# =============================== FLB ==================================== #
def nk_FLB_split(kx, ky, T, h, mu, t_up, tp_up, t_dn, tp_dn, a=1.0, n_target=None):
    if n_target is None:
        raise ValueError("nk_FLB_split requires n_target for initial guess")

    coskx = np.cos(kx)
    cosky = np.cos(ky)
    e_up = 2.0 * t_up * (coskx + cosky) + 4.0 * tp_up * coskx * cosky - h/2.0 - mu
    e_down = 2.0 * t_dn * (coskx + cosky) + 4.0 * tp_dn * coskx * cosky + h/2.0 - mu

    initial_guess = [n_target/2.0, n_target/2.0]
    res = minimize(
        func_FLB,
        initial_guess,
        args=(e_up, e_down, T),
        jac=func_deriv_FLB,
        constraints=cons_FLB,
        method='SLSQP',
        options={'eps': 1e-10, 'disp': False}
    )
    if not res.success:
        raise RuntimeError(f"FLB minimize failed at k=({kx},{ky}): {res.message}")
    return res.x


def n_tot_FLB_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a=1.0, n_target=None):
    sum_up = 0.0
    sum_dn = 0.0
    for i in range(L):
        kx = i * 2.0 * math.pi / L
        for j in range(L):
            ky = j * 2.0 * math.pi / L
            n_up, n_dn = nk_FLB_split(kx, ky, T, h, mu, t_up, tp_up, t_dn, tp_dn, a, n_target)
            sum_up += n_up
            sum_dn += n_dn
    return sum_up / (L*L), sum_dn / (L*L)


def density_FLB_split(mu, T, h, L, n_target, t_up, tp_up, t_dn, tp_dn, a=1.0):
    n_up, n_dn = n_tot_FLB_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, n_target)
    return n_up + n_dn - n_target


def solve_mu_FLB_split(n_target, T, L, h, t_up, tp_up, t_dn, tp_dn, a=1.0, mu_low=-80, mu_high=80):
    f = lambda mu: density_FLB_split(mu, T, h, L, n_target, t_up, tp_up, t_dn, tp_dn, a)
    return root_scalar(f, bracket=[mu_low, mu_high], method='brentq').root


# =============================== NFL ==================================== #
def nk_NFL_split(kx, ky, T, h, mu, t_up, tp_up, t_dn, tp_dn, a=1.0, n_target=None):
    if n_target is None:
        raise ValueError("nk_NFL_split requires n_target for initial guess")

    coskx = np.cos(kx)
    cosky = np.cos(ky)
    e_up = 2.0 * t_up * (coskx + cosky) + 4.0 * tp_up * coskx * cosky - h/2.0 - mu
    e_down = 2.0 * t_dn * (coskx + cosky) + 4.0 * tp_dn * coskx * cosky + h/2.0 - mu

    initial_guess = [n_target/2.0, n_target/2.0]
    res = minimize(
        func_NFL,
        initial_guess,
        args=(e_up, e_down, T),
        jac=func_deriv_NFL,
        constraints=cons_NFL,
        method='SLSQP',
        options={'eps': 1e-10, 'disp': False}
    )
    if not res.success:
        raise RuntimeError(f"NFL minimize failed at k=({kx},{ky}): {res.message}")
    return res.x


def n_tot_NFL_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a=1.0, n_target=None):
    sum_up = 0.0
    sum_dn = 0.0
    for i in range(L):
        kx = i * 2.0 * math.pi / L
        for j in range(L):
            ky = j * 2.0 * math.pi / L
            n_up, n_dn = nk_NFL_split(kx, ky, T, h, mu, t_up, tp_up, t_dn, tp_dn, a, n_target)
            sum_up += n_up
            sum_dn += n_dn
    return sum_up / (L*L), sum_dn / (L*L)


def density_NFL_split(mu, T, h, L, n_target, t_up, tp_up, t_dn, tp_dn, a=1.0):
    n_up, n_dn = n_tot_NFL_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, n_target)
    return n_up + n_dn - n_target


def solve_mu_NFL_split(n_target, T, L, h, t_up, tp_up, t_dn, tp_dn, a=1.0, mu_low=-80, mu_high=80):
    f = lambda mu: density_NFL_split(mu, T, h, L, n_target, t_up, tp_up, t_dn, tp_dn, a)
    return root_scalar(f, bracket=[mu_low, mu_high], method='brentq').root


# ============================ One-state and Entropy ====================== #
def one_state_split(dist, T, n_target, h, kx, ky, t_up, tp_up, t_dn, tp_dn, a=1.0):
    L = len(kx)
    if dist == "FD":
        mu = solve_mu_FD_split(n_target, T, L, h, t_up, tp_up, t_dn, tp_dn, a)
        n_up, n_dn = n_tot_FD_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, n_target)
    elif dist == "FLB":
        mu = solve_mu_FLB_split(n_target, T, L, h, t_up, tp_up, t_dn, tp_dn, a)
        n_up, n_dn = n_tot_FLB_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, n_target)
    elif dist == "NFL":
        mu = solve_mu_NFL_split(n_target, T, L, h, t_up, tp_up, t_dn, tp_dn, a)
        n_up, n_dn = n_tot_NFL_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, n_target)
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


def compute_total_entropy_split(T, h, mu, L, t_up, tp_up, t_dn, tp_dn, a, distribution, n_target=None):
    kx = np.linspace(0, 2 * math.pi / a, L, endpoint=False)
    ky = kx.copy()

    total_entropy = 0.0
    for i in range(L):
        for j in range(L):
            kxi = kx[i]
            kyj = ky[j]
            if distribution == "FD":
                n_up, n_dn = nk_FD_split(kxi, kyj, T, h, mu, t_up, tp_up, t_dn, tp_dn, a, n_target)
            elif distribution == "FLB":
                n_up, n_dn = nk_FLB_split(kxi, kyj, T, h, mu, t_up, tp_up, t_dn, tp_dn, a, n_target)
            elif distribution == "NFL":
                n_up, n_dn = nk_NFL_split(kxi, kyj, T, h, mu, t_up, tp_up, t_dn, tp_dn, a, n_target)
            else:
                raise ValueError(distribution)

            total_entropy += entropy_k_point(n_up, n_dn, T, distribution)

    weight = (2*math.pi / L)**2 / (2*math.pi)**2
    return total_entropy * weight


