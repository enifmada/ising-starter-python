import sys
import os
import time
import csv
import click
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm #fancy progress bar generator
from ising import run_ising #import run_ising function from ising.py
import multiprocessing as mp

def calculate_and_save_values(Msamp, Esamp, spin, num_analysis, index, temp, data_filename, corr_filename):
    try:
        #calculate statistical values
        Msamp_analysis = Msamp[-num_analysis:]
        Esamp_analysis = Esamp[-num_analysis:]
        Msamp_analysis_sq = np.square(Msamp_analysis)
        Esamp_analysis_sq = np.square(Esamp_analysis)
        M_mean = np.average(Msamp_analysis)
        E_mean = np.average(Esamp_analysis)
        c_v = 1.0/temp*(np.average(Esamp_analysis_sq)-E_mean**2)
        chi = np.average(Msamp_analysis_sq)-M_mean**2
        M_std = np.std(Msamp_analysis)
        E_std = np.std(Esamp_analysis)
        data_array = [M_mean,M_std,E_mean,E_std,c_v,chi]

        #write data to CSV file
        header_array = ['Temperature','Magnetization Mean','Magnetization Std Dev','Energy Mean','Energy Std Dev',"Heat Capacity","Susceptibility"]
        append_data_to_file(data_filename, header_array) if index == 0 else None
        append_data_to_file(data_filename, data_array, temp)

        #get correlation function
        corr = compute_autocorrelation(spin)

        #write correlation function to CSV file
        header_array = ['Temperature','K','Spatial Spin Correlation']
        append_data_to_file(corr_filename, header_array) if index == 0 else None
        [append_data_to_file(corr_filename, corr_value, temp) for corr_value in corr]

        return True

    except:
        logging.error("Temp=" + str(temp) + ": Statistical Calculation Failed. No Data Written.")
        sys.exit()

#simulation options (enter python main.py --help for details)
@click.command()
@click.option('--t_min', default=2.22, help='Minimum Temperature (inclusive)', type=float)
@click.option('--t_max', default=2.32001, help='Maximum Temperature (exclusive)', type=float)
@click.option('--t_step', default=0.005, help='Temperature Step Size', type=float)
@click.option('--t_anneal', default=20.0, help="Starting Annealing Temperature", type=float)
@click.option("--anneal_boolean", default=True, help="Anneal or not?", type=bool)
#anneal_boolean should always be true unless testing stuff related to annealing

@click.option('--n', default=10, help='Lattice Size (NxN)',type=int)
@click.option('--num_steps', default=18000, help='Total Number of Steps',type=int)
@click.option('--num_analysis', default=12000, help='Number of Steps used in Analysis',type=int)
@click.option('--num_burnin', default=5000, help='Total Number of Burnin Steps',type=int)

@click.option('--j', default=1.0, help='Interaction Strength',type=float)
@click.option('--b', default=0.0, help='Applied Magnetic Field',type=float)
@click.option('--b_anneal', default=1.0, help="Starting Annealing Field", type=float)
@click.option('--flip_prop', default=0.1, help='Proportion of Spins to Consider Flipping per Step',type=float)

@click.option('--output', default='data', help='Directory Name for Data Output',type=str)
@click.option('--plots', default=False, help='Turn Automatic Plot Creation Off or On',type=bool)

@click.option('--processes', default=4, help='',type=int)


def ising(t_min,t_max,t_step,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,plots,t_anneal,b_anneal,anneal_boolean,output,processes):
    length=0
    t_move = t_min
    while t_move < t_max:
        length = length + 1
        t_move = t_move + t_step
    data_filename, corr_filename = initialize_simulation(n,num_steps,num_analysis,num_burnin,output,j,b,flip_prop,t_anneal,b_anneal,length)
    run_processes(processes,t_min,t_max,t_step,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,plots,t_anneal,b_anneal,anneal_boolean,data_filename,corr_filename)
    print('\n\nSimulation Finished! Data written to '+ data_filename)

def initialize_simulation(n,num_steps,num_analysis,num_burnin,output,j,b,flip_prop,t_anneal,b_anneal,length):
    check_step_values(num_steps, num_analysis, num_burnin)
    data_filename, corr_filename = get_filenames(output)
    write_sim_parameters(data_filename,corr_filename,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,t_anneal,b_anneal,length)
    print('\nSimulation Started! Data will be written to ' + data_filename + '\n')
    return data_filename, corr_filename

def run_processes(processes,t_min,t_max,t_step,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,plots,t_anneal,b_anneal,anneal_boolean,data_filename,corr_filename):

    T = get_temp_array(t_min, t_max, t_step)
    pool = mp.Pool(processes=processes)
    [pool.apply_async(run_simulation, args=(index,temp,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,plots,t_anneal,b_anneal,anneal_boolean,data_filename,corr_filename,),callback = print_result) for index,temp in enumerate(T)]
    pool.close()
    pool.join()


def print_result(temp):
    print("Temp {} Done".format(temp))

def run_simulation(index,temp,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,plots,t_anneal,b_anneal,anneal_boolean,data_filename,corr_filename):
    # if plots:
    #     #initialize vars for plotting values
    #     temp_arr, M_mean_arr, E_mean_arr, M_std_arr, E_std_arr = [],[],[],[],[]

    try:
        #run the Ising model
        Msamp, Esamp, spin = run_ising(n,temp,num_steps,num_burnin,flip_prop,j,b,t_anneal,b_anneal,anneal_boolean, disable_tqdm=True)
        #plt.plot(Esamp[:20000])
        #plt.show()

        #get and save statistical values
        if calculate_and_save_values(Msamp,Esamp,spin,num_analysis,index,temp,data_filename,corr_filename):

            if plots:
                #for plotting
                M_mean, E_mean, M_std, E_std = get_plot_values(temp,Msamp,Esamp,num_analysis)
                temp_arr.append(temp)
                M_mean_arr.append(M_mean)
                E_mean_arr.append(E_mean)
                M_std_arr.append(M_std)
                E_std_arr.append(E_std)

    except KeyboardInterrupt:
        print("\n\nProgram Terminated. Good Bye!")
        sys.exit()

    except:
        logging.error("Temp="+str(temp)+": Simulation Failed. No Data Written")

    # if plots:
    #     plot_graphs(temp_arr, M_mean_arr, E_mean_arr, M_std_arr, E_std_arr)

def get_plot_values(temp,Msamp,Esamp,num_analysis): #only for plotting at end
    try:
        M_mean = np.average(Msamp[-num_analysis:])
        E_mean = np.average(Esamp[-num_analysis:])
        M_std = np.std(Msamp[-num_analysis:])
        E_std = np.std(Esamp[-num_analysis:])
        return M_mean, E_mean, M_std, E_std
    except:
        logging.error("Temp={0}: Error getting plot values".format(temp))
        return False

def plot_graphs(temp_arr,M_mean_arr,M_std_arr,E_mean_arr,E_std_arr): #plot graphs at end
    plt.figure(1)
    plt.ylim(0,1)
    plt.errorbar(temp_arr, np.absolute(M_mean_arr), yerr=M_std_arr, uplims=True, lolims=True,fmt='o')
    plt.xlabel('Temperature')
    plt.ylabel('Magnetization')
    plt.figure(2)
    plt.errorbar(temp_arr, E_mean_arr, yerr=E_std_arr, fmt='o')
    plt.xlabel('Temperature')
    plt.ylabel('Energy')
    plt.show()

def check_step_values(num_steps,num_analysis,num_burnin): #simulation size checks and exceptions
    if (num_burnin > num_steps):
        raise ValueError('num_burning cannot be greater than available num_steps. Exiting simulation.')

    if (num_analysis > num_steps - num_burnin):
        raise ValueError('num_analysis cannot be greater than available num_steps after burnin. Exiting simulation.')

def get_filenames(dirname): #make data folder if doesn't exist, then specify filename
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        data_filename = os.path.join(dirname,'data_'+str(time.strftime("%Y%m%d-%H%M%S"))+".csv")
        corr_filename = os.path.join(dirname,'corr_'+str(time.strftime("%Y%m%d-%H%M%S"))+".csv")
        #Write simulation parameters to file
        return data_filename, corr_filename
    except:
        raise ValueError('Directory name not valid. Exiting simulation.')

def write_sim_parameters(data_filename,corr_filename,n,num_steps,num_analysis,num_burnin,j,b,flip_prop,t_anneal,b_anneal,length):
    try:
        with open(data_filename,'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            #Write simulations parameters to CSV file
            writer.writerow(['Lattice Size (NxN)','Total Steps/Temp','Steps/Temp Used in Analysis','Burnin Steps','Interaction Strength','Applied Mag Field','Spin Prop', "Starting Anneal Temp", "Starting Anneal Field", "Number of Temps"])
            writer.writerow([n,num_steps,num_analysis,num_burnin,j,b,flip_prop,t_anneal,b_anneal,length])
            writer.writerow([])
        with open(corr_filename,'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            #Write simulations parameters to CSV file
            writer.writerow(['Lattice Size (NxN)','Total Steps','Steps Used in Analysis','Burnin Steps','Interaction Strength','Applied Mag Field','Spin Prop', "Starting Anneal Temp", "Starting Anneal Field", "Number of Temps"])
            writer.writerow([n,num_steps,num_analysis,num_burnin,j,b,flip_prop,t_anneal,b_anneal, length])
            writer.writerow([])
    except:
        logging.error('Could not save simulation parameters. Exiting simulation')
        sys.exit()

def get_temp_array(t_min,t_max,t_step):
    if (t_min > t_max):
        raise ValueError('T_min cannot be greater than T_max. Exiting Simulation')
    try:
        T = np.arange(t_min,t_max,t_step).tolist()
        return T
    except:
        raise ValueError('Error creating temperature array. Exiting simulation.')

def compute_autocorrelation(spin):
    n = len(spin)
    corr_array = []
    for k in range(1,int(n/2)):
        col_mean, row_mean = spin.mean(axis=0),spin.mean(axis=1)
        #compute r values for rows and cols
        r_col = [np.multiply(spin[j,:]-col_mean,spin[(j+k)%n,:]-col_mean) for j in range(1,n)]
        r_row = [np.multiply(spin[:,j]-row_mean,spin[:,(j+k)%n]-row_mean) for j in range(1,n)]
        #normalize r values
        r_col = np.divide(r_col,float(n))
        r_row = np.divide(r_row,float(n))
        #calculate corr for k and add it to array
        corr = (r_col.mean() + r_row.mean())/2.0
        corr_array.append([k,corr])
    return corr_array

def append_data_to_file(filename,data_array,temp=False):
    try:
        with open(filename,'a') as csv_file: #appends to existing CSV File
            writer = csv.writer(csv_file, delimiter=',', lineterminator='\n')
            if temp:
                writer.writerow([temp]+data_array)
            else:
                writer.writerow(data_array)

    except:
        logging.error("Temp={0}: Error Writing to File".format(temp))

if __name__ == "__main__":
    print("\n2D Ising Model Simulation\n")
    ising()
