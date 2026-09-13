import numpy as np
from integrators import rk4


def dynamics_rimless_wheel(t, state, params):
    """Compute the time derivative of the state (i.e. state dynamics) for a rimless wheel."""
    gravity = params["gravity"]
    length = params["length"]

    theta = state[0]
    theta_dot = state[1]

    acceleration = gravity/length * np.sin(theta)  # Simplified model for a rimless wheel

    state_derivative = np.array([theta_dot, acceleration])
    return state_derivative


def apply_impact(state, params):
    """Check for impact within a timestep while the rimless wheel is 
    rolling either forward (next spoke) or backward (previous spoke), 
    as well as apply new angular velocity and shift coordinates if necessary."""
    theta = state[0]
    theta_dot = state[1]
    alpha = np.pi / params["number_of_spokes"]
    gamma = params["gamma"]

    # Impact-event guard: next spoke has reached the ground 
    # (forward rolling impact)
    if theta >= (gamma + alpha) and theta_dot > 0:
        new_theta_dot = theta_dot * np.cos(2 * alpha)
        new_theta = gamma - alpha
        return np.array([new_theta, new_theta_dot]), True
    
    # Impact-event guard: previous spoke has reached the ground 
    # (backward rolling impact)
    elif theta <= (gamma - alpha) and theta_dot < 0:
        new_theta_dot = theta_dot * np.cos(2 * alpha)
        new_theta = gamma + alpha
        return np.array([new_theta, new_theta_dot]), True
    
    else:
        return state, False


def integrate_with_impact(state, t, timestep, params):
    """Integrates the rimless wheel dynamics over a single timestep and checks 
    for impact events. If an impact is detected, the time of the impact is 
    refined using a bisection procedure. The post-impact state, impact time, 
    and a flag indicating whether an impact occurred are returned."""
    gamma = params["gamma"]
    alpha = np.pi / params["number_of_spokes"]

    old_state = state.copy()
    trial_state = rk4(old_state, timestep, dynamics_rimless_wheel, t, params)

    # Extract the old and trial angles for impact detection.
    old_theta = old_state[0]
    new_theta = trial_state[0]
    new_theta_dot = trial_state[1]

    # Determine if a forward or backward impact occurred
    forward_impact = (old_theta < (gamma + alpha)
        and new_theta >= (gamma + alpha) and new_theta_dot > 0)

    backward_impact = (old_theta > (gamma - alpha) 
        and new_theta <= (gamma - alpha) and new_theta_dot < 0)

    # If no impact occurred, return the trial state as the new state
    if not (forward_impact or backward_impact):
        return trial_state, None, False, None

    if forward_impact:
        event_guard = gamma + alpha
    else:
        event_guard = gamma - alpha

    # Bisection bounds for impact time
    impact_time_lower = 0.0
    impact_time_upper = timestep

    # Perform bisection to find the precise impact time
    for _ in range(30):
        impact_time_midpoint = 0.5 * (impact_time_lower + impact_time_upper)
        midpoint_state = rk4(old_state, impact_time_midpoint, 
            dynamics_rimless_wheel, t, params)
        midpoint_theta = midpoint_state[0]

        # Check if the midpoint state has crossed the impact event guard
        if forward_impact:
            if midpoint_theta >= event_guard:
                impact_time_upper = impact_time_midpoint
            else:
                impact_time_lower = impact_time_midpoint
        else:
            if midpoint_theta <= event_guard:
                impact_time_upper = impact_time_midpoint
            else:
                impact_time_lower = impact_time_midpoint

    # Use the midpoint of the final bisection interval as the impact time
    timestep_to_impact = 0.5 * (impact_time_lower + impact_time_upper)

    impact_state = rk4(old_state, timestep_to_impact, 
        dynamics_rimless_wheel, t, params)

    # Force theta exactly onto the impact event guard
    impact_state[0] = event_guard

    # Apply the impact to get the post-impact state
    post_impact_state, did_impact = apply_impact(impact_state, params)
    impact_time = t + timestep_to_impact

    # Continuing integrating for the remainder of the timestep
    remaining_timestep = timestep - timestep_to_impact
    if remaining_timestep > 0:
        final_state = rk4(post_impact_state, remaining_timestep, 
            dynamics_rimless_wheel, impact_time, params)
    else:
        final_state = post_impact_state.copy()

    # Return the exact post-impact state and precise impact time
    return final_state, impact_time, did_impact, post_impact_state

    
def calculate_energy(state, params):
    """Compute energies for a state ``(2,)`` or trajectory ``(2, N)``."""
    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]

    theta = state[0]  # indexes entire row "vectorized" if state is (2, N)
    theta_dot = state[1]

    kinetic_energy = 0.5 * mass * (length ** 2) * (theta_dot ** 2)
    potential_energy = mass * gravity * length * np.cos(theta)

    return kinetic_energy, potential_energy


def find_steady_state_velocity(params):
    """Return pre- and post-impact steady-state angular velocities."""
    gravity = params["gravity"]
    length = params["length"]
    gamma = params["gamma"]
    alpha = np.pi / params["number_of_spokes"]

    # Pre-impact steady state angular velocity from energy balance derivation
    theta_dot_minus_steady = np.sqrt((2 * gravity * (np.cos(gamma - 
        alpha) - np.cos(gamma + alpha))) / (length * 
        (np.sin(2 * alpha) ** 2)))
    # Post-impact steady state angular velocity
    theta_dot_plus_steady = theta_dot_minus_steady * (
        np.cos(2 * alpha))

    return theta_dot_minus_steady, theta_dot_plus_steady


def classify_attractor(initial_state, params, timestep, sim_time,
                       min_impacts, steady_velocity_tolerance,
                       rest_velocity_tolerance, theta_dot_plus_steady):
    """Classify the attractor type for the rimless wheel given an initial state.
    Attractor type:
        0 = unclassified
        1 = rest
        2 = steady rolling
    """
    state = initial_state.copy()
    t = 0.0
    impact_velocities = []
    num_timesteps = int(np.ceil(sim_time / timestep))

    # Simulation loop for the rimless wheel dynamics
    for step in range(num_timesteps):
        if t >= sim_time:
            break
        state, impact_time, did_impact, post_impact_state = (
            integrate_with_impact(state, t, timestep, params))
        t = t + timestep

        if did_impact:
            post_impact_velocity = state[1]
            impact_velocities.append(post_impact_velocity)
            # Check for attractor 1 or "rest" by seeing if the recent impact 
            # velocities are close to zero
            if len(impact_velocities) >= min_impacts:
                recent = np.asarray(impact_velocities[-min_impacts:])
                max_recent_velocity = np.max(np.abs(recent))
                if max_recent_velocity < rest_velocity_tolerance:
                    return 1

            # Check for attractor 2 or "steady rolling" by seeing if the recent 
            # impact velocities are all positive and close to each other, as
            # well as close to the theoretical post-impact steady state 
            # angular velocity
            if len(impact_velocities) >= min_impacts:
                recent = np.asarray(impact_velocities[-min_impacts:])
                all_positive = np.all(recent > 0)
                fixed_point_error = np.max(np.abs(recent - 
                    theta_dot_plus_steady))
                impact_spread = (np.max(recent) - np.min(recent))
                if (all_positive and fixed_point_error < 
                    steady_velocity_tolerance and impact_spread
                    < steady_velocity_tolerance):
                    return 2

    # Final check for attractors after the simulation loop has ended
    if len(impact_velocities) >= min_impacts:
        recent = np.asarray(impact_velocities[-min_impacts:])
        max_recent_velocity = np.max(np.abs(recent))
        if (max_recent_velocity < rest_velocity_tolerance):
            return 1

    if (len(impact_velocities) >= min_impacts):
        recent = np.asarray(impact_velocities[-min_impacts:])
        fixed_point_error = np.max(np.abs(recent - 
            theta_dot_plus_steady))
        impact_spread = (np.max(recent) - np.min(recent))
        if (np.all(recent > 0) and fixed_point_error 
            < steady_velocity_tolerance and impact_spread 
            < steady_velocity_tolerance):
            return 2
    return 0


def find_limit_cycle(params, timestep, theta_dot_plus_steady, 
                     theta_dot_minus_steady):
    """Compute limit cycle while the rimless wheel enters steady state rolling."""
    gamma = params["gamma"]
    alpha = np.pi / params["number_of_spokes"]

    state = np.array([gamma - alpha, theta_dot_plus_steady])
    trajectory = [state.copy()]
    times = [0.0]
    t = 0.0

    while True:
        # Step forward
        state, impact_time, did_impact, post_impact_state = (
            integrate_with_impact(state, t, timestep, params))
        t = t + timestep
        
        if did_impact:
            # If an impact occurred, append the pre-impact state at the boundary (gamma + alpha)
            # and break before the coordinate reset takes effect
            trajectory.append(np.array([gamma + alpha, state[1] / (
                np.cos(2 * alpha))]))
            break
        else:
            trajectory.append(state.copy())
            
        times.append(t)

    trajectory = np.asarray(trajectory)
    return (trajectory[:, 0], trajectory[:, 1], theta_dot_plus_steady,
        theta_dot_minus_steady)


def calculate_roa(params, theta_dot_min, theta_dot_max,
                  num_grid, roa_timestep, roa_sim_time):
    """Calculate the RoA grid for a given parameter set."""
    gamma = params["gamma"]
    alpha = np.pi / params["number_of_spokes"]

    theta_dot_minus_steady, theta_dot_plus_steady = ( 
        find_steady_state_velocity(params))

    theta_values = np.linspace(gamma - alpha, gamma + alpha, num_grid)
    theta_dot_values = np.linspace(theta_dot_min, theta_dot_max,
                                   num_grid)
    Angle, Angular_Velocity = np.meshgrid(theta_values, theta_dot_values)

    roa = np.zeros(Angle.shape, dtype=int)

    for i in range(Angle.shape[0]):
        for j in range(Angle.shape[1]):
            initial_state = np.array([Angle[i, j],
                Angular_Velocity[i, j]])
            roa[i, j] = classify_attractor(initial_state, params,
                roa_timestep, roa_sim_time, min_impacts=5,
                steady_velocity_tolerance=3e-3, 
                rest_velocity_tolerance=3e-3, 
                theta_dot_plus_steady=theta_dot_plus_steady)
            # Progress information
            completed = i + 1
            percentage = (completed / Angle.shape[0]) * 100
            print(f"RoA Progress: {percentage:.2f}% completed", end="\r")

    return theta_values, theta_dot_values, roa


def simulate_one_step_from_section(theta_dot, params, timestep):
    """Start immediately after an impact and simulate until the
    next impact.  Return the post-impact angular velocity there.
    """
    alpha = np.pi / params["number_of_spokes"]
    gamma = params["gamma"]

    # Immediately after a forward impact, theta = gamma - alpha
    state = np.array([gamma - alpha, theta_dot])
    t = 0.0

    # Simulate until the next impact
    for step in range(1000000):
        state, impact_time, did_impact, post_impact_state = (
            integrate_with_impact(state, t, timestep, params))
        t = t + timestep
        if did_impact:
            return post_impact_state[1]

    raise RuntimeError("No impact detected during one-step simulation.")


def calculate_floquet_multiplier(params, timestep, delta):
    """Calculate the Floquet multiplier from Poincare section."""
    theta_dot_minus_steady, theta_dot_plus_steady = (
        find_steady_state_velocity(params))

    theta_dot_minus = theta_dot_plus_steady - delta
    theta_dot_plus = theta_dot_plus_steady + delta

    P_minus = simulate_one_step_from_section(
        theta_dot_minus, params, timestep)
    P_plus = simulate_one_step_from_section(
        theta_dot_plus, params, timestep)

    # Approximate the derivative of the Poincare map at the fixed point 
    # using a finite difference
    floquet_multiplier = (P_plus - P_minus) / (2 * delta)

    return floquet_multiplier, P_minus, P_plus


def calculate_floquet_multiplier_sweep(params, timestep, deltas):
    """Calculate Floquet multipliers across an array of delta values."""
    floquet_values = []
    for i, delta in enumerate(deltas):
        floquet_multiplier, P_minus, P_plus = (
            calculate_floquet_multiplier(params, timestep, delta))
        floquet_values.append(floquet_multiplier)

        # Progress information
        completed = i + 1
        percentage = (completed / len(deltas)) * 100
        print(f"Floquet Sweep Progress: {percentage:.2f}% completed", end="\r")
    return np.array(floquet_values)


def calculate_inclination_sweep(params, gamma_values, timestep,
                                roa_timestep, roa_sim_time, 
                                theta_dot_min, theta_dot_max, 
                                num_grid, gamma_roa_values=None):
    """Calculate Floquet multipliers and RoA fractions 
    across inclinations."""

    gamma_values = np.asarray(gamma_values)
    if gamma_roa_values is None:
        gamma_roa_values = []

    gamma_floquet = []
    gamma_rolling_fraction = []
    gamma_rest_fraction = []
    roa_gamma_results = {}

    # Loop over each inclination value to compute Floquet multipliers 
    # and RoA fractions
    for i, gamma_value in enumerate(gamma_values):
        sweep_params = params.copy()
        sweep_params["gamma"] = gamma_value
        # Calculate Floquet multiplier
        floquet_multiplier, P_minus, P_plus = (
            calculate_floquet_multiplier(sweep_params, timestep=timestep,
            delta=1e-5))
        gamma_floquet.append(floquet_multiplier)

        # Calculate RoA
        theta_values, theta_dot_values, roa_grid = calculate_roa(
            sweep_params, theta_dot_min=theta_dot_min, 
            theta_dot_max=theta_dot_max, num_grid=num_grid, 
            roa_timestep=roa_timestep, roa_sim_time=roa_sim_time)

        print(f"\nInclination {i + 1}/{len(gamma_values)} "
        f"completed: gamma = {gamma_value:.3f}")

        total_points = roa_grid.size

        rolling_fraction = np.sum(roa_grid == 2) / total_points
        rest_fraction = np.sum(roa_grid == 1) / total_points

        gamma_rolling_fraction.append(rolling_fraction)
        gamma_rest_fraction.append(rest_fraction)

        # Store selected RoA grids
        if gamma_value in gamma_roa_values:
            roa_gamma_results[gamma_value] = (theta_values,
                theta_dot_values, roa_grid)

    return (np.asarray(gamma_floquet), np.asarray(gamma_rolling_fraction),
        np.asarray(gamma_rest_fraction), roa_gamma_results)


def calculate_spoke_sweep(params, spoke_values, timestep,
                          roa_timestep, roa_sim_time, theta_dot_min, theta_dot_max, num_grid,
                          spoke_roa_values=None):
    """Calculate Floquet multipliers and RoA fractions 
    across spoke angles."""

    spoke_values = np.asarray(spoke_values)
    if spoke_roa_values is None:
        spoke_roa_values = []

    spoke_floquet = []
    spoke_rolling_fraction = []
    spoke_rest_fraction = []
    roa_spoke_results = {}

    # Loop over each spoke angle value to compute Floquet multipliers 
    # and RoA fractions
    for i, spoke_value in enumerate(spoke_values):
        sweep_params = params.copy()
        sweep_params["number_of_spokes"] = spoke_value
        # Calculate Floquet multiplier
        floquet_multiplier, P_minus, P_plus = (
            calculate_floquet_multiplier(sweep_params, timestep=timestep,
            delta=1e-5))
        spoke_floquet.append(floquet_multiplier)

        # Calculate RoA
        theta_values, theta_dot_values, roa_grid = calculate_roa(
            sweep_params, theta_dot_min=theta_dot_min, 
            theta_dot_max=theta_dot_max, num_grid=num_grid, 
            roa_timestep=roa_timestep, roa_sim_time=roa_sim_time)

        print(f"\nSpoke {i + 1}/{len(spoke_values)} "
        f"completed: number_of_spokes = {spoke_value}")

        total_points = roa_grid.size

        rolling_fraction = np.sum(roa_grid == 2) / total_points
        rest_fraction = np.sum(roa_grid == 1) / total_points

        spoke_rolling_fraction.append(rolling_fraction)
        spoke_rest_fraction.append(rest_fraction)

        # Store selected RoA grids
        if spoke_value in spoke_roa_values:
            roa_spoke_results[spoke_value] = (theta_values,
                theta_dot_values, roa_grid)

    return (np.asarray(spoke_floquet), np.asarray(spoke_rolling_fraction),
        np.asarray(spoke_rest_fraction), roa_spoke_results)