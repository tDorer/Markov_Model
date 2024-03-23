import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sci_con

# Defining the main class for the modeling
class model:
    # Constructor
    def __init__(self,initial_conc, free_energies, transition_free_energies, labels = 0,
                 enzyme_conc = 0.0001, k0 = 4_000_000, dt = 0.025, unit = "kBT", temperature = 310):
        self.dt     = dt
        self.__time = [0]                            # time array for simulating the reactions
        self.__conc = [np.array(initial_conc)]       # Vector of initial concentration
        self.__number_states = len(initial_conc)     # The number of states the model has
        # Arange labels for plot
        if labels==0:
            labels=[]
            for i in range(self.__number_states):
                labels.append("State"+str(i))
        self.__labels=labels
        # Calculate transition rates
        ## Calculating Free Energy differences from the given state values and the barrier values
        differences = []
        for i in range(self.__number_states):
            diff = []
            for j in range(self.__number_states):
                if transition_free_energies[i][j]!=0:
                    diff.append(transition_free_energies[i][j]-free_energies[i])
                else:
                    diff.append(0)
            differences.append(diff)
        ## Transfer free energy differences into given unit
        if unit not in ["kBT", "kJ/mol", "eV"]:
            raise ValueError("Unknown unit, please provide one of the following inputs: kJ/mol, eV, kBT.")
        elif unit == "kJ/mol":
            differences = np.array(differences)/temperature/sci_con.k/sci_con.Avogadro*1000
        elif unit == "eV":
            differences = np.array(differences)/temperature/sci_con.k*sci_con.e
        ## Assigning transition probabilities
        transition_probabilities = k0*enzyme_conc*np.exp(-np.array(differences))
        ## Assigning probabilities to the diagonal
        for i in range(self.__number_states):
            transition_probabilities[i][i] = -sum(transition_probabilities[i])
        self.__transition_probabilities = transition_probabilities.T

    # Method to simulate the state model forward in time
    def forward(self,time_steps, save_every = 10):
        for t in range(time_steps):
            self.__time.append(self.__time[-1] + self.dt*time_steps*save_every)
            conc = self.__conc[-1] + self.dt*np.matmul(self.__transition_probabilities,self.__conc[-1])
            if save_every-1!=0:
                for i in range(save_every-1):
                    conc = conc + self.dt*np.matmul(self.__transition_probabilities,conc)                
            self.__conc.append(conc)

    # Method to plot the saved concentrations vs. the time
    ## Give if the values should be plotted in a figure that exists, give the plt.axis object as axis.
    def plot(self, axis="none", title = False, xlabel = "time [s]", ylabel = "relative Concentration [%]", legend=False, grid="both", fontsize = 13, fontname = "ARIAL", fontweight = "bold"):
        if axis == "none":
            fig = plt.figure(figsize=(10,5), dpi=320)
            axis = fig.add_subplot(111)
        for i in range(self.__number_states):
            axis.plot(self.__time,np.array(self.__conc)[:,i],label=self.__labels[i])
        if xlabel != "none":
            axis.set_xlabel(xlabel, fontsize = fontsize, fontname = fontname, fontweight = fontweight)
        if ylabel != "none":
            axis.set_ylabel(ylabel, fontsize = fontsize, fontname = fontname, fontweight = fontweight)
        if grid in ["both","x","y"]:
            axis.grid(True, which=grid)
        if title!=False:
            axis.set_title(title, fontsize = fontsize, fontname = fontname, fontweight = fontweight)
        if legend!=False:
            plt.legend(loc=legend, fontsize = fontsize)
