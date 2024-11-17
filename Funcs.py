import numpy as np

def fl_hat(r_LH, r_HL):
    """
    Calculate the equilibrium probability of a cell being in the L state.

    Args:
        r_LH (float): Rate of transition from L to H.
        r_HL (float): Rate of transition from H to L.

    Returns:
        float: Probability of a cell being in the L state.
    """
    return r_HL / (r_LH + r_HL)

def fh_hat(r_LH, r_HL):
    """
    Calculate the equilibrium probability of a cell being in the H state.

    Args:
        r_LH (float): Rate of transition from L to H.
        r_HL (float): Rate of transition from H to L.

    Returns:
        float: Probability of a cell being in the H state.
    """
    return 1 - fl_hat(r_LH, r_HL)

def h_out(N_H, r_HL):
    """
    Calculate the number of H cells transitioning to L state.

    Args:
        N_H (int): Number of H cells.
        r_HL (float): Rate of transition from H to L.

    Returns:
        int: Number of H cells transitioning to L state.
    """
    return np.random.binomial(N_H, r_HL)

def l_out(N_L, r_LH):
    """
    Calculate the number of L cells transitioning to H state.

    Args:
        N_L (int): Number of L cells.
        r_LH (float): Rate of transition from L to H.

    Returns:
        int: Number of L cells transitioning to H state.
    """
    return np.random.binomial(N_L, r_LH)

def m_L(L, N_L, mu_L):
    """
    Calculate the number of mutations in L cells.

    Args:
        L (int): Total number of cells.
        N_L (int): Number of L cells.
        mu_L (float): Mutation rate for L cells.

    Returns:
        int: Number of mutations in L cells.
    """
    return np.random.binomial(L * N_L, mu_L)

def m_H(L, N_H, mu_H):
    """
    Calculate the number of mutations in H cells.

    Args:
        L (int): Total number of cells.
        N_H (int): Number of H cells.
        mu_H (float): Mutation rate for H cells.

    Returns:
        int: Number of mutations in H cells.
    """
    return np.random.binomial(L * N_H, mu_H)

def m_tot(L, N_H, N_L, mu_H, mu_L):
    """
    Calculate the total number of mutations in both H and L cells.

    Args:
        L (int): Total number of cells.
        N_H (int): Number of H cells.
        N_L (int): Number of L cells.
        mu_H (float): Mutation rate for H cells.
        mu_L (float): Mutation rate for L cells.

    Returns:
        int: Total number of mutations in both H and L cells.
    """
    return m_H(L, N_H, mu_H) + m_L(L, N_L, mu_L)

def init_col(p_H, N_0):
    """
    Initialize a colony with a given draw probability of H cells.

    Args:
        p_H (float): Probability of a cell being in the H state.
        N_0 (int): Initial number of cells.

    Returns:
        tuple: Number of H cells and L cells in the colony.
    """
    N_H = np.random.binomial(N_0, p_H)
    N_L = N_0 - N_H
    return N_H, N_L

def next_state(N_H, N_L, r_HL, r_LH):
    """
    Calculate the next state of the colony based on transition rates.

    Args:
        N_H (int): Number of H cells.
        N_L (int): Number of L cells.
        r_HL (float): Rate of transition from H to L.
        r_LH (float): Rate of transition from L to H.

    Returns:
        tuple: Updated number of H cells and L cells in the colony.
    """
    hout = h_out(N_H, r_HL)
    lout = l_out(N_L, r_LH)
    N_H_next = N_H - hout + lout
    N_L_next = N_L - lout + hout
    return 2 * N_H_next, 2 * N_L_next