#%% Import
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Go back one directory to import custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import potts_sim  # Import the custom module
from potts_utils import parse_graph
from cim_sim import run_cim_from_graph
from test_bench import execute_model, plot_phase_with_wraparound

# Apply theme
sns.set_theme(style="white")
#plt.style.use('tableau-colorblind10')
sns.set_palette("colorblind")
plt.rc('font', family='Liberation sans')

#%% Set up the simulation parameters
 
# Default parameters
T = 100         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps

num_states = 3 # number of states for Potts model (q=k)

noise_factor = 1e-4
seed = 2

# Load a coupling graph
#file_path = "graphs/band/band50_3_antiferro.col"
#file_path = "graphs/gset/G5.col"
file_path = "../graphs/g05/g05_20.0.col"
num_vertices, num_edges, edges, opt_cut_dict, opt_energy_dict, mu_max = parse_graph(file_path)
opt_cut = opt_cut_dict.get(num_states, 0)

edges_per_vertex = num_edges / num_vertices  # average number of edges per vertex

res_dict = {}  # Dictionary to store results for each model

#%% Run NEC

T = 1000         # total simulation time
dt = 1e-2       # time step
num_steps = int(np.floor(T / dt))  # number of time steps

polynomial_order = 3 # n=3 used in original paper, but n>=q is required for amplitude bounding
alpha_rate = 4e-2 # rate of change of alpha (NEC model)
r_target = 1 # target radius for NEC model
initial_alpha_arr = -mu_max * np.ones(num_vertices) # initial alpha values for NEC model

# constant gamma
#gamma_schedule = np.ones(num_steps) * 0.3

# linear gamma schedule
gamma_schedule  = np.linspace(0, 1, num_steps)
gamma_schedule *= 3 # scale by factor

res_dict["NEC"] = execute_model("NEC", potts_sim.run_nec, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
    polynomial_order, alpha_rate, r_target,
    initial_alpha_arr, gamma_schedule
)

#%% Run q-PDC model
T = 10         # total simulation time
dt = 1e-3       # time step
num_steps = int(np.floor(T / dt))  # number of time steps

dampening = 1 + edges_per_vertex
gamma_th = (256/27)**(1/4)

poly_order = 5 # order of polynomial model (Polynomial model)
beta_schedule  = np.ones(num_steps) * 1/dampening 
gamma_schedule = np.linspace(0,gamma_th*5, num_steps)
gamma_schedule *= 1

amplitude_seed = 1

res_dict["q-PDC"] = execute_model("q-PDC", potts_sim.run_polynomial, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
    amplitude_seed,
    poly_order,
    list(beta_schedule), list(gamma_schedule)
)

#%% Run polynomial model

T = 1000         # total simulation time
dt = 1e-2       # time step
num_steps = int(np.floor(T / dt))  # number of time steps

poly_order = 3 # order of polynomial model (Polynomial model)

beta_schedule = np.linspace(1/mu_max, 1/mu_max + 0.2, num_steps) # beta schedule
gamma_schedule = beta_schedule * 1 # gamma schedule

amplitude_seed = 0

res_dict["Polynomial"] = execute_model("Polynomial", potts_sim.run_polynomial, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
    amplitude_seed,
    poly_order,
    list(beta_schedule), list(gamma_schedule)
)


#%% Run Sigmoid model

T = 1000         # total simulation time
dt = 1e-2       # time step
num_steps = int(np.floor(T / dt))  # number of time steps

alpha = -10.0 # alpha value for sigmoid model

beta_schedule  = np.linspace((1-alpha)/mu_max, (1-alpha)/mu_max + 2, num_steps) # beta schedule (Polynomial and Sigmoid models)
gamma_schedule = beta_schedule * 2 # gamma schedule (Polynomial, Sigmoid, and Fixed-Amplitude models)

res_dict["Sigmoid"] = execute_model("Sigmoid", potts_sim.run_sigmoid, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, 2,
    alpha, list(beta_schedule), list(gamma_schedule)
)

#%% Run Fixed-Amplitude model

T = 1000         # total simulation time
dt = 1e-2       # time step
num_steps = int(np.floor(T / dt))  # number of time steps

gamma_schedule = np.linspace(0,2,num_steps)

res_dict["Fixed-Amplitude"] = execute_model("Fixed-Amplitude", potts_sim.run_fixed_amplitude, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
    list(gamma_schedule)
)

#%% Run CIM model

T = 1000          # total simulation time
dt = 1e-2        # time step
num_steps = int(np.floor(T / dt))  # number of time steps

# CIM specific parameters
zeta = 0.5       # empirical rescaling factor
B_num_vertices = 20  # B/A/num_    vertices ratio for soft constraints 
B = B_num_vertices / num_vertices  # B value for CIM model
alpha = 0.9      # parameter for tanh nonlinearity
beta_schedule = np.linspace(0, 0.4, num_steps)  # time-dependent annealing schedule

# Execute the CIM model
res_dict["CIM"] = execute_model("CIM", run_cim_from_graph, T, dt, num_vertices, num_states, edges, opt_cut, noise_factor, seed,
    alpha,
    list(beta_schedule),
    B, zeta
)

# %% Dynamics plot

# Parameters
num_states = 3

# Calculate theta values for gridlines
theta_values = [(2 * np.pi / num_states) * (j - np.floor(num_states / 2))
                for j in range(num_states)]
# add -pi and pi
theta_values = np.array(theta_values + [-np.pi, np.pi])
theta_labels = [r"$-\frac{2\pi}{3}$", r"$0$", r"$\frac{2\pi}{3}$", r"$-\pi$", r"$\pi$"]

# Create figure
fig = plt.figure(figsize=(10, 10))

# Simplified outer GridSpec: just [top], [bottom]
outer = GridSpec(2, 1, height_ratios=[2,1], hspace=0.3)

# Top section: 3 rows (amp, phase, cut) x 4 columns
top = outer[0].subgridspec(3, 4, height_ratios=[1, 1, 1*2/3], hspace=0.35, wspace=-0.3)

# Bottom section: 2 rows (main, cut) x 2 columns (wide)
bottom = outer[1].subgridspec(2, 2, height_ratios=[1, 1*2/3], hspace=0.40, wspace=-0.3)

# Models and labels
models = ["NEC", "q-PDC", "Polynomial", "Sigmoid"]
display_names = ["NEC", r'$q$-PDC', "Polynomial PM\n(this work)", "Sigmoid PM\n(this work)"]
letters = ['a', 'b', 'c', 'd']

max_num_spins = 100 # Maximum number of spins to plot

# Top models: amplitude, phase, cut in 3 rows
for idx, (model_key, display_name, letter) in enumerate(zip(models, display_names, letters)):
    aspect_ratio = 1
    # Get model results
    model_res = res_dict[model_key]
    cont_states = model_res["continuous_states"]
    
    # Downsample if needed
    if cont_states.shape[0] > 1000:
        idxs = np.linspace(0, cont_states.shape[0] - 1, 1000, dtype=int)
        cont_states = cont_states[idxs]
    
    # Create time array
    T = model_res["dt"] * model_res["num_steps"]
    t = np.linspace(0, T, cont_states.shape[0])

    # Amplitude row (row 0)
    axA = fig.add_subplot(top[0, idx])
    axA.set_box_aspect(aspect_ratio)
    axA.text(-0.2, 1.10, f"({letter}1)", transform=axA.transAxes,
             fontsize=12, fontweight="bold")
    axA.set_title(display_name, pad=10)
    
    # Plot amplitudes
    if np.iscomplexobj(cont_states):
        amp = np.abs(cont_states)
        for i in range(min(max_num_spins, cont_states.shape[1])):  # Plot first 5 spins to avoid clutter
            axA.plot(t, amp[:, i], lw=1)
    else:
        for i in range(min(max_num_spins, cont_states.shape[1])):
            axA.plot(t, cont_states[:, i], lw=1)
    
    if idx == 0:
        axA.set_ylabel("Spin amplitudes")
    axA.tick_params(axis='both', which='both', direction='out', 
                   bottom=True, top=False, left=True, right=False,
                   labelbottom=False)
    axA.grid(False)
    #axA.set_xlim(t.min(), t.max())

    # Phase row (row 1)
    axP = fig.add_subplot(top[1, idx])
    axP.set_box_aspect(aspect_ratio)
    axP.text(-0.2, 1.10, f"({letter}2)", transform=axP.transAxes,
             fontsize=12, fontweight="bold")
    
    # Plot phases if complex
    if np.iscomplexobj(cont_states):
        phase = np.angle(cont_states)
        plot_phase_with_wraparound(
            axP, t, phase,
            max_num_to_plot=max_num_spins,
            lw=1
        )
    else:
        axP.text(0.5, 0.5, "Real-valued\nspins", ha='center', va='center', transform=axP.transAxes)
    
    axP.set_ylim(-np.pi, np.pi)
    # remove x‐axis label and ticks (replaced by cut plot below)
    axP.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    if idx == 0:
        axP.set_ylabel("Spin phases")
    else:
        axP.tick_params(axis='y', labelleft=False)
    
    # Add horizontal gridlines and ticks
    for theta in theta_values:
        axP.axhline(theta, color='gray', linestyle='--', lw=0.5)
    axP.set_yticks(theta_values)
    axP.set_yticklabels(theta_labels)
    axP.grid(axis='x', visible=False)
    #axP.set_xlim(t.min(), t.max())

    # Cut value row (row 2)
    axC = fig.add_subplot(top[2, idx])
    axC.set_box_aspect(aspect_ratio*2/3)
    axC.text(-0.2, 1.10, f"({letter}3)", transform=axC.transAxes,
             fontsize=12, fontweight="bold")
    cut = np.array(model_res["cut_value"])
    if cut.size > 1000:
        idxs = np.linspace(0, cut.size - 1, 1000, dtype=int)
        cut = cut[idxs]
        t_cut = np.linspace(0, T, cut.size)
    else:
        t_cut = np.linspace(0, T, cut.size)
    axC.plot(t_cut, cut, lw=1)
    axC.axhline(opt_cut, color='red', linestyle='--', label='Optimum')
    axC.legend(loc='lower right')
    axC.set_ylim(50, opt_cut+3)  # Set y-axis range
    axC.set_yticks(np.arange(50, opt_cut+5, 10))  # Set y-ticks 
    if idx == 0:
        axC.set_ylabel("Cut value\n(Max-3-Cut)")
    else:
        axC.tick_params(axis='y', labelleft=False)
    axC.set_xlabel("Time")
    axC.tick_params(axis='both', which='both', direction='out',
                   bottom=True, top=False, left=True, right=False)
    axC.grid(False)
    #axC.set_xlim(t_cut.min(), t_cut.max())

# Bottom models: 2 wide columns, 2 rows (main and cut)
bottom_models = [
    ("Fixed-Amplitude", r'$q$-SHIL', "e", "Spin phases"),
    ("CIM", "Reference IM", "f", "Spin amplitudes")
]

for j, (model_key, display_name, letter, ylabel) in enumerate(bottom_models):
    aspect_ratio = 0.7  # Aspect ratio for main plots
    # Main plot (row 0)
    ax = fig.add_subplot(bottom[0, j])
    ax.set_box_aspect(aspect_ratio)
    ax.text(-0.2, 1.10, f"({letter}1)", transform=ax.transAxes,
            fontsize=12, fontweight="bold")
    ax.set_title(display_name, pad=10)

    # Get model results
    model_res = res_dict[model_key]
    cont_states_full = model_res["continuous_states"]

    # Create full time array and extract inset data before downsampling
    T = model_res["dt"] * model_res["num_steps"]
    t_full = np.linspace(0, T, cont_states_full.shape[0])
    inset_tmax = 3
    inset_mask = t_full <= inset_tmax
    t_inset_full = t_full[inset_mask]
    cont_states_inset = cont_states_full[inset_mask]

    # Downsample for main plot if needed
    cont_states = cont_states_full
    if cont_states.shape[0] > 1000:
        idxs = np.linspace(0, cont_states.shape[0] - 1, 1000, dtype=int)
        cont_states = cont_states[idxs]

    # Create time array for main plot
    t = np.linspace(0, T, cont_states.shape[0])

    # For q-SHIL (Fixed-Amplitude), plot phase
    if model_key == "Fixed-Amplitude":
        if np.iscomplexobj(cont_states):
            phase = np.angle(cont_states)
            plot_phase_with_wraparound(
                ax, t, phase,
                max_num_to_plot=max_num_spins,
                lw=1
            )
        else:
            ax.text(0.5, 0.5, "Real-valued\nspins", ha='center', va='center', transform=ax.transAxes)
        
        ax.set_ylim(-np.pi, np.pi)
        # Add horizontal gridlines and ticks
        for theta in theta_values:
            ax.axhline(theta, color='gray', linestyle='--', lw=0.5)
        ax.set_yticks(theta_values)
        ax.set_yticklabels(theta_labels)
    
    # For Reference Ising machine (CIM), plot amplitude
    else:
        if np.iscomplexobj(cont_states):
            amp = np.abs(cont_states)
            for i in range(min(max_num_spins, cont_states.shape[1])):
                ax.plot(t, amp[:, i], lw=1)
        else:
            for i in range(min(max_num_spins, cont_states.shape[1])):
                ax.plot(t, cont_states[:, i], lw=1)

        # Add inset showing t=0 to inset_tmax (full resolution)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_inset = inset_axes(ax, width="35%", height="45%", loc='upper right',
                              bbox_to_anchor=(0, 0, 0.99, 0.82), bbox_transform=ax.transAxes)
        if np.iscomplexobj(cont_states_inset):
            amp_inset = np.abs(cont_states_inset)
            for i in range(min(max_num_spins, amp_inset.shape[1])):
                ax_inset.plot(t_inset_full, amp_inset[:, i], lw=0.5)
        else:
            for i in range(min(max_num_spins, cont_states_inset.shape[1])):
                ax_inset.plot(t_inset_full, cont_states_inset[:, i], lw=0.5)
        inset_yrange = np.max(np.abs(amp_inset)) * 1.1 if np.iscomplexobj(cont_states_inset) else np.max(np.abs(cont_states_inset)) * 1.1
        ax_inset.set_xlim(0, inset_tmax)
        ax_inset.set_xticks([0, inset_tmax])
        ax_inset.set_ylim(-inset_yrange, inset_yrange)
        ax_inset.tick_params(axis='both', which='both', labelsize=7, pad=-3)
        ax_inset.set_xlabel("Time", fontsize=7, labelpad=-2)

    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='both', direction='out',
                  bottom=True, top=False, left=True, right=False,
                  labelbottom=False)
    ax.grid(False)
    #ax.set_xlim(t.min(), t.max())

    # Cut value plot (row 1)
    axC = fig.add_subplot(bottom[1, j])
    axC.set_box_aspect(aspect_ratio*2/3)
    axC.text(-0.2, 1.15, f"({letter}2)", transform=axC.transAxes,
             fontsize=12, fontweight="bold")
    cut = np.array(res_dict[model_key]["cut_value"])
    if cut.size > 1000:
        idxs = np.linspace(0, cut.size - 1, 1000, dtype=int)
        cut = cut[idxs]
        t_cut = np.linspace(0, T, cut.size)
    else:
        t_cut = np.linspace(0, T, cut.size)
    axC.plot(t_cut, cut, lw=1)
    axC.axhline(opt_cut, color='red', linestyle='--', label='Optimum')
    axC.legend(loc='lower right')
    axC.set_ylim(50, opt_cut+3)  # Set y-axis range
    axC.set_yticks(np.arange(50, opt_cut+5, 10))  # Set y-ticks 
    
    axC.set_ylabel("Cut value\n(Max-3-Cut)")
    axC.tick_params(axis='both', which='both', direction='out',
                   bottom=True, top=False, left=True, right=False)
    axC.grid(False)
    #axC.set_xlim(t_cut.min(), t_cut.max())
    # remove x‐ticks
    axC.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    axC.set_xlabel("Time")

plt.tight_layout()
plt.savefig("dynamics_figure.pdf", bbox_inches='tight')
plt.show()

#%% Similar figure, but individual figure for each model
top_panel_size = (2.5, 6.7)
bottom_panel_size = (2.5, 4.2)

top_figures = [
    ("NEC", "NEC"),
    ("q-PDC", r"$q$-PDC"),
    ("Polynomial", "Polynomial PM (this work)"),
    ("Sigmoid", "Sigmoid PM (this work)")
]

for model_key, title in top_figures:
    data = res_dict[model_key]
    cont = data["continuous_states"]
    if cont.shape[0] > 1000:
        sel = np.linspace(0, cont.shape[0] - 1, 1000, dtype=int)
        cont = cont[sel]
    total_time = data["dt"] * data["num_steps"]
    time_axis = np.linspace(0, total_time, cont.shape[0])

    fig = plt.figure(figsize=top_panel_size)
    gs = GridSpec(3, 1, height_ratios=[1, 1, 2/3], hspace=0.35)
    ax_amp = fig.add_subplot(gs[0])
    ax_phase = fig.add_subplot(gs[1])
    ax_cut = fig.add_subplot(gs[2])
    fig.suptitle(title, y=0.94)

    if np.iscomplexobj(cont):
        amplitudes = np.abs(cont)
        for col in range(min(max_num_spins, amplitudes.shape[1])):
            ax_amp.plot(time_axis, amplitudes[:, col], lw=1)
    else:
        for col in range(min(max_num_spins, cont.shape[1])):
            ax_amp.plot(time_axis, cont[:, col], lw=1)
    ax_amp.set_ylabel("Spin amplitudes")
    ax_amp.tick_params(labelbottom=False)

    if np.iscomplexobj(cont):
        phases = np.angle(cont)
        plot_phase_with_wraparound(ax_phase, time_axis, phases, max_num_to_plot=max_num_spins, lw=1)
        ax_phase.set_ylim(-np.pi, np.pi)
        for theta in theta_values:
            ax_phase.axhline(theta, color="gray", linestyle="--", lw=0.5)
        ax_phase.set_yticks(theta_values)
        ax_phase.set_yticklabels(theta_labels)
    else:
        ax_phase.text(0.5, 0.5, "Real-valued spins", ha="center", va="center", transform=ax_phase.transAxes)
    ax_phase.set_ylabel("Spin phases")
    ax_phase.tick_params(labelbottom=False)

    cut_series = np.asarray(data["cut_value"])
    if cut_series.size > 1000:
        sel = np.linspace(0, cut_series.size - 1, 1000, dtype=int)
        cut_series = cut_series[sel]
        cut_time = np.linspace(0, total_time, cut_series.size)
    else:
        cut_time = np.linspace(0, total_time, cut_series.size)
    ax_cut.plot(cut_time, cut_series, lw=1)
    ax_cut.axhline(opt_cut, color="red", linestyle="--", label="Optimum")
    ax_cut.set_ylabel("Cut value\n(Max-3-Cut)")
    ax_cut.set_xlabel("Time")
    ax_cut.legend(loc="lower right")
    ax_cut.set_ylim(50, opt_cut + 3)
    ax_cut.set_yticks(np.arange(50, opt_cut + 5, 10))

    fig.tight_layout()
    fig.savefig(f"dynamics_{model_key.lower()}.png", bbox_inches="tight")
    plt.close(fig)

bottom_figures = [
    ("Fixed-Amplitude", r"$q$-SHIL", True),
    ("CIM", "Reference IM", False)
]

for model_key, title, is_phase_plot in bottom_figures:
    data = res_dict[model_key]
    cont = data["continuous_states"]
    if cont.shape[0] > 1000:
        sel = np.linspace(0, cont.shape[0] - 1, 1000, dtype=int)
        cont = cont[sel]
    total_time = data["dt"] * data["num_steps"]
    time_axis = np.linspace(0, total_time, cont.shape[0])

    fig = plt.figure(figsize=bottom_panel_size)
    gs = GridSpec(2, 1, height_ratios=[1, 2/3], hspace=0.40)
    ax_main = fig.add_subplot(gs[0])
    ax_cut = fig.add_subplot(gs[1])
    fig.suptitle(title, y=0.94)

    if is_phase_plot and np.iscomplexobj(cont):
        phases = np.angle(cont)
        plot_phase_with_wraparound(ax_main, time_axis, phases, max_num_to_plot=max_num_spins, lw=1)
        ax_main.set_ylim(-np.pi, np.pi)
        for theta in theta_values:
            ax_main.axhline(theta, color="gray", linestyle="--", lw=0.5)
        ax_main.set_yticks(theta_values)
        ax_main.set_yticklabels(theta_labels)
        ax_main.set_ylabel("Spin phases")
    elif is_phase_plot:
        ax_main.text(0.5, 0.5, "Real-valued spins", ha="center", va="center", transform=ax_main.transAxes)
        ax_main.set_ylabel("Spin phases")
    elif np.iscomplexobj(cont):
        amplitudes = np.abs(cont)
        for col in range(min(max_num_spins, amplitudes.shape[1])):
            ax_main.plot(time_axis, amplitudes[:, col], lw=1)
        ax_main.set_ylabel("Spin amplitudes")
    else:
        for col in range(min(max_num_spins, cont.shape[1])):
            ax_main.plot(time_axis, cont[:, col], lw=1)
        ax_main.set_ylabel("Spin amplitudes")
    ax_main.tick_params(labelbottom=False)

    cut_series = np.asarray(data["cut_value"])
    if cut_series.size > 1000:
        sel = np.linspace(0, cut_series.size - 1, 1000, dtype=int)
        cut_series = cut_series[sel]
        cut_time = np.linspace(0, total_time, cut_series.size)
    else:
        cut_time = np.linspace(0, total_time, cut_series.size)
    ax_cut.plot(cut_time, cut_series, lw=1)
    ax_cut.axhline(opt_cut, color="red", linestyle="--", label="Optimum")
    ax_cut.set_ylabel("Cut value\n(Max-3-Cut)")
    ax_cut.set_xlabel("Time")
    ax_cut.legend(loc="lower right")
    ax_cut.set_ylim(50, opt_cut + 3)
    ax_cut.set_yticks(np.arange(50, opt_cut + 5, 10))

    fig.tight_layout()
    fig.savefig(f"dynamics_{model_key.lower()}.png", bbox_inches="tight")
    plt.close(fig)

