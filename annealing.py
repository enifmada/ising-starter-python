import numpy as np

def T_anneal(T, t_anneal, alpha, step, num_burnin):
    if step < num_burnin:
        T_a = t_anneal*np.power(alpha,step)
    else:
        T_a = T
    return T_a

def B_anneal(B, b_anneal, beta, step, num_burnin):
    if step < num_burnin:
        B_a = b_anneal*np.power(beta,step)
    else:
        B_a = B
    return B_a
