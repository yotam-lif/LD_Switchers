import numpy as np

def fl_hat(r_LH, r_HL):
    return r_HL / (r_LH + r_HL)

def fh_hat(r_LH, r_HL):
    return 1 - fl_hat(r_LH, r_HL)

def h_out(N_H, r_HL):
    return np.random.binomial(N_H, r_HL)

def l_out(N_L, r_LH):
    return np.random.binomial(N_L, r_LH)

def m_L(L, N_L, mu_L):
    return np.random.binomial(L * N_L, mu_L)

def m_H(L, N_H, mu_H):
    return np.random.binomial(L * N_H, mu_H)

def m_tot(L, N_H, N_L, mu_H, mu_L):
    return m_H(L, N_H, mu_H) + m_L(L, N_L, mu_L)

def init_col(p_H, N_0):
    N_H = np.random.binomial(N_0, p_H)
    N_L = N_0 - N_H
    return N_H, N_L

def next_state(N_H, N_L, r_HL, r_LH):
    hout = h_out(N_H, r_HL)
    lout = l_out(N_L, r_LH)
    N_H_next = N_H - hout + lout
    N_L_next = N_L - lout + hout
    return N_H_next, N_L_next