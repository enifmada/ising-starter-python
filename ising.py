import numpy as np
from scipy import signal
from annealing import B_anneal, T_anneal

try:
    __IPYTHON__
except:
    from tqdm import tqdm

def run_ising(N,T,num_steps,num_burnin,flip_prop,J,B,t_anneal,anneal_boolean):

    # Description of parameters:
    # N = Grid Size
    # T = Temperature (normalized to k_B = 1)
    # num_steps = Number of steps to run in total (including burnin steps)
    # num_burnin = Number of steps to use for the burnin process.  This isn't
    #	used in this code but you might need it if you change this to try and
    #	get better convergence.
    # J = Interaction strength
    # B = Applied magnetic field
    # flip_prop = Total ratio of spins to possibly flip per step

    # Initialize variables
    M,E = 0,0 # Magnetization and Energy Initial Values
    Msamp, Esamp = [],[] #Arrays to hold magnetization and energy values

    # We obtain the sum of nearest neighbors by convoluting
    #   this matrix with the spin matrix
    conv_mat = np.matrix('0 1 0; 1 0 1; 0 1 0')
    alpha = np.exp(np.log(T/t_anneal)/num_burnin)
    print(alpha)
    # Generate a random initial configuration
    spin = np.random.choice([-1,1],(N,N))

    try:
        __IPYTHON__
        steps = range(num_steps)
    except:
        steps = tqdm(range(num_steps))

    # Evolve the system
    for step in steps:

        try:
            __IPYTHON__
        except:
            steps.set_description("Working on T = %.2f" % T)
            steps.refresh() # to show immediately the update

        #implement annealing in annealing.py file
        if anneal_boolean:
            T_step = T_anneal(T, t_anneal, alpha, step, num_burnin)
            B_step = B_anneal(B, step, num_steps, num_burnin)
        else:
            T_step = T
            B_step = B

        #Calculating the total spin of neighbouring cells
        neighbors = signal.convolve2d(spin,conv_mat,mode='same',boundary='wrap')

        #Sum up our variables of interest, normalize by N^2
        M = float(np.sum(spin))/float(N**2)
        Msamp.append(M)

        #Divide by two because of double counting
        E = float(-J*(np.sum((spin*neighbors)))/2.0 - float(B_step)*M)/float(N**2)
        Esamp.append(E)

        #Calculate the change in energy of flipping a spin
        DeltaE = 2.0 * (J*(spin*neighbors) + float(B_step)*spin)

        #Calculate the transition probabilities
        p_trans = np.where(DeltaE >= 0.0, np.exp(-1.0*DeltaE/float(T_step)),1.0)
        #If DeltaE is positive, calculate the Boltzman flipping probability.
        #If not, assign a transition probability of 1

        #Decide which transitions will occur
        transitions = [[-1 if (cell>np.random.rand() and flip_prop>np.random.rand()) else 1 for cell in row] for row in p_trans]
        #Perform the transitions
        spin = spin*transitions

    return Msamp, Esamp, spin
