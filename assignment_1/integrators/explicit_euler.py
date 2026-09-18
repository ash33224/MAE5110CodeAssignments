def explicit_euler(state, timestep, dynamics_model, t, params):
    """Computes a single explicit Euler step."""
    return state + timestep * dynamics_model(t, state, params)