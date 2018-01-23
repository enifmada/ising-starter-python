import numpy as np

def T_anneal(T, t_anneal, alpha, step, num_burnin):
    if step < num_burnin:
        T_a = t_anneal*np.power(alpha,step)
    else:
        T_a = T
    return T_a

def B_anneal(B, ii, num_steps, num_burnin):

    #implement annealing code here

    B_a = B

    return float(B_a)
