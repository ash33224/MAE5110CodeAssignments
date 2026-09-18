def rk4(state, timestep, dynamics_model, t, params):
    """Computes a single Runge-Kutta step."""
    k1 = dynamics_model(t, state, params)
    k2 = dynamics_model(t + 1/2 * timestep, state + 1/2 * timestep * k1, params)
    k3 = dynamics_model(t + 1/2 * timestep, state + 1/2 * timestep * k2, params)
    k4 = dynamics_model(t + timestep, state + timestep * k3, params) 
    return state + (timestep / 6) * (k1 + 2 * k2 + 2 * k3 + k4)