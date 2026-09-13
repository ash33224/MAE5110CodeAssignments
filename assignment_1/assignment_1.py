import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from models import rimless_wheel as model
from integrators import rk4


# Basic simulation of a rimless wheel

params = {
    "gravity": 9.81,  # gravity (m/s^2)
    "mass": 1.0,  # mass of ball (kg)
    "length": 1.0,  # spoke length (m)
    "number_of_spokes": 8,  # number of spokes
    "gamma": 0.08,  # downhill inclination of the ground (rad)
}
gravity = params["gravity"]
mass = params["mass"]
length = params["length"]
gamma = params["gamma"]
alpha = np.pi / params["number_of_spokes"]

# Set-up
initial_state = np.array([0.2, 2])

timestep = 1e-5
sim_time = 5.0

# Initial condition must be between gamma - alpha and gamma + alpha to ensure 
# the rimless wheel starts within a single spoke-to-spoke step
if initial_state[0] >= params["gamma"] + alpha or (
    initial_state[0] <= params["gamma"] - alpha):
    raise ValueError(
        f"Starting theta is out of bounds. "
        f"Choose an angle between {params['gamma'] - alpha:.3f} rad "
        f"and {params['gamma'] + alpha:.3f} rad.")

# Pre- and Post-impact steady state angular velocity 
theta_dot_minus_steady, theta_dot_plus_steady = (
    model.find_steady_state_velocity(params))

def simulate_rimless_wheel(timestep, sim_time, initial_state, params):
    n_timesteps = int(np.ceil(sim_time / timestep)) + 1
    time_traj = np.arange(n_timesteps) * timestep
    state_traj = np.zeros((2, n_timesteps))
    state_traj[:, 0] = initial_state

    post_impact_velocities = []
    impact_times = []

    # Simulation loop with rk4
    for step, t in enumerate(time_traj[:-1]):

        # Integrate rimless wheel dynamics over one timestep 
        # and check for impact
        state_traj[:, step + 1], impact_time, did_impact, post_impact_state = (
            model.integrate_with_impact(state_traj[:, step], t, 
                timestep, params))
        if did_impact:
            post_impact_velocities.append(post_impact_state[1])
            impact_times.append(impact_time)

    return state_traj, time_traj, np.array(post_impact_velocities), \
           np.array(impact_times)

state_traj, time_traj, post_impact_velocities, impact_times = \
    simulate_rimless_wheel(timestep, sim_time, initial_state, params)
kinetic_energy, potential_energy = model.calculate_energy(
    state_traj, params)

# Plot the energies
plt.figure()
plt.plot(time_traj, potential_energy, label="Potential energy")
plt.plot(time_traj, kinetic_energy, label="Kinetic energy")
plt.plot(time_traj, potential_energy + kinetic_energy, label="Total energy")
plt.xlabel("Time (s)")  
plt.ylabel("Energy (J)")
plt.title(f"Rimless Wheel Energy (dt = {timestep:.5f})")
plt.legend()
plt.tight_layout()
plt.show()

# Plot the phase portrait
plt.figure()
plt.plot(state_traj[0, :], state_traj[1, :])
plt.xlabel(r"Angle $\theta$ (rad)")
plt.ylabel(r"Angular velocity "
           r"$\dot{\theta}$ (rad/s)")
plt.title(f"Rimless Wheel Phase Portrait (dt = {timestep:.5f})")
plt.axvline(gamma - alpha, color="gray", linestyle="--", 
            label="Impact boundaries")
plt.axvline(gamma + alpha, color="gray", linestyle="--")
plt.text(gamma - alpha, 0.5, r"$\gamma-\alpha$", color="gray", 
         ha="left", va="top")
plt.text(gamma + alpha, 0.5, r"$\gamma+\alpha$", color="gray",
         ha="right", va="top")
plt.axhline(0, color="black", linewidth=0.8)
plt.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# Plot region of attraction
# Simulation parameters for RoA grid scan
num_grid = 51
roa_timestep = 5e-4 
roa_sim_time = 20.0
theta_dot_min = -0.6
theta_dot_max = 1.8

theta_values, theta_dot_values, roa = model.calculate_roa(params,
    theta_dot_min=theta_dot_min, theta_dot_max=theta_dot_max, 
    num_grid=num_grid, roa_timestep=roa_timestep, 
    roa_sim_time=roa_sim_time)

Angle, Angular_Velocity = np.meshgrid(theta_values, theta_dot_values)
roa_cmap = ListedColormap([
    "lightgray",    # 0 = unclassified
    "royalblue",    # 1 = rest
    "darkorange",   # 2 = rolling
    ])
plt.figure(figsize=(9, 6))
plt.contourf(Angle, Angular_Velocity, roa, levels=[-0.5, 0.5, 1.5, 2.5],
              cmap=roa_cmap)
# Calculate limit cycle
theta, theta_dot, theta_dot_plus_steady, theta_dot_minus_steady = (
    model.find_limit_cycle(params, roa_timestep, theta_dot_plus_steady,
                            theta_dot_minus_steady))
# Overlay limit cycle
plt.plot(theta, theta_dot, color="black", linewidth=2.5, 
         label="Limit cycle")
plt.xlabel(r"Initial angle $\theta_0$ (rad)")
plt.ylabel(r"Initial angular velocity "
            r"$\dot{\theta}_0$ (rad/s)")
plt.title(f"Regions of Attraction for Rimless Wheel: "
          f"$\\gamma = {gamma:.3f}$, "f"$\\alpha = {alpha:.3f}$")
legend_elements = [Patch(label="Rest", color="royalblue"), 
                   Patch(label="Steady State", color="darkorange"),
                   Patch(label="Unclassified", color="lightgray"),
                   Patch(label="Limit cycle", color="black")]
plt.legend(handles=legend_elements)
plt.grid(True, linestyle=":", alpha=0.4)
plt.tight_layout()
plt.show()

# Plot Poincare section/one-dimensional step-to-step return map
# post_impact_velocities[k] = angular velocity immediately
# after the kth spoke contact
theta_dot_k = post_impact_velocities[:-1]
theta_dot_next = post_impact_velocities[1:]

plt.figure(figsize=(7, 6))
# Return map data
plt.plot(theta_dot_k, theta_dot_next, "o-", color="darkorange",
         markersize=5, linewidth=1, label="Return map")

# Identity line: theta_dot_{k+1} = theta_dot_k
theta_dot_min = min(np.min(theta_dot_k), np.min(theta_dot_next))
theta_dot_max = max(np.max(theta_dot_k), np.max(theta_dot_next))
theta_dot_range = np.linspace(theta_dot_min, theta_dot_max, 200)

plt.plot(theta_dot_range, theta_dot_range, "k--",
         linewidth=1.5, label=r"Identity: $\dot{\theta}_{k+1}=\dot{\theta}_k$")
plt.plot(theta_dot_plus_steady, theta_dot_plus_steady,
         "ro", markersize=9,
         label=fr"Theoretical fixed point = {theta_dot_plus_steady:.3f}")
plt.xlabel(r"Angular velocity at crossing $k$, $\dot{\theta}_k^+$ (rad/s)")
plt.ylabel(r"Angular velocity at crossing $k+1$, $\dot{\theta}_{k+1}^+$ (rad/s)")
plt.title("Step-to-Step Poincare Return Map")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Compute Floquet multiplier from the Poincare return map
# Define an array of delta values spanning multiple orders of magnitude
deltas = np.logspace(-7, -1, 40)

# Compute Floquet multipliers for each delta
floquet_sweep = model.calculate_floquet_multiplier_sweep(
    params, timestep, deltas)

# Plot Floquet multiplier vs. delta 
plt.figure(figsize=(7, 5))
plt.semilogx(deltas, floquet_sweep, "o-", color="purple", markersize=4)
plt.xlabel(r"Perturbation size ($\Delta \dot{\theta}$)", fontsize=12)
plt.ylabel("Computed Floquet Multiplier", fontsize=12)
plt.title("Floquet Multiplier Convergence Analysis")
plt.grid(True, which="both", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

# Sweep the inclinations (gamma)
gamma_values = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 
                         0.16, 0.18, 0.20])
gamma_roa_values = gamma_values

(gamma_floquet, gamma_rolling_fraction, gamma_rest_fraction, 
    roa_gamma_results) = model.calculate_inclination_sweep(params,
    gamma_values, timestep=1e-5, roa_timestep=roa_timestep,
    roa_sim_time=roa_sim_time, theta_dot_min=theta_dot_min,
    theta_dot_max=theta_dot_max, num_grid=num_grid,
    gamma_roa_values=gamma_roa_values)
    
# Plot Floquet multiplier vs inclination
plt.figure(figsize=(8, 5))
plt.plot(gamma_values, gamma_floquet, "o-", color="darkorange",
         linewidth=2, markersize=6)
plt.axhline(1, color="black", linestyle="--", linewidth=1,
            label=r"$\lambda=1$")
plt.axhline(-1, color="gray", linestyle="--", linewidth=1,
            label=r"$\lambda=-1$")
plt.axhline(0, color="gray", linewidth=0.8)
plt.xlabel(r"Inclination $\gamma$ (rad)")
plt.ylabel(r"Floquet multiplier $\lambda$")
plt.title("Floquet Multiplier vs. Inclination")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Plot RoA fractions vs inclination
plt.figure(figsize=(8, 5))
plt.plot(gamma_values, gamma_rolling_fraction, "o-", color="darkorange",
    label="Steady rolling")
plt.plot(gamma_values, gamma_rest_fraction, "s-", color="royalblue",
          label="Rest")
plt.xlabel(r"Inclination $\gamma$ (rad)")
plt.ylabel("Fraction of RoA grid")
plt.title("Region of Attraction vs. Inclination")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Sweep number of spokes
spoke_values = np.array([6, 7, 8, 9, 10, 11, 12])
spoke_roa_values = spoke_values

(spoke_floquet, spoke_rolling_fraction, spoke_rest_fraction, 
    roa_spoke_results) = (model.calculate_spoke_sweep(params, 
    spoke_values, timestep=timestep, roa_timestep=roa_timestep, 
    roa_sim_time=roa_sim_time, theta_dot_min=theta_dot_min,
    theta_dot_max=theta_dot_max, num_grid=51,
    spoke_roa_values=spoke_roa_values))

# Plot Floquet multiplier vs spoke count
plt.figure(figsize=(8, 5))
plt.plot(spoke_values, spoke_floquet, "o-", color="darkorange",
         linewidth=2, markersize=6)
plt.axhline(1, color="black", linestyle="--", linewidth=1,
            label=r"$\lambda=1$")
plt.axhline(-1, color="gray", linestyle="--", linewidth=1,
            label=r"$\lambda=-1$")
plt.axhline(0, color="gray", linewidth=0.8)
plt.xlabel(r"Number of spokes $N$")
plt.ylabel(r"Floquet multiplier $\lambda$")
plt.title("Floquet Multiplier vs. Number of Spokes")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# Plot RoA fractions vs number of spokes
plt.figure(figsize=(8, 5))
plt.plot(spoke_values, spoke_rolling_fraction, "o-", color="darkorange",
    label="Steady rolling")
plt.plot(spoke_values, spoke_rest_fraction, "s-", color="royalblue",
          label="Rest")
plt.xlabel(r"Number of spokes $N$")
plt.ylabel("Fraction of RoA grid")
plt.title("Region of Attraction vs. Number of Spokes")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()