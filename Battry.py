import numpy as np
import cvxpy as cp

# Energy System Parameters
initial_battery_soc = 0.5  # Initial State of Charge (SOC) of the battery
battery_capacity = 100  # Battery capacity in kWh
solar_capacity = 50  # Solar panel capacity in kWh

# Simulation Parameters
num_steps = 24  # Number of time steps (hours) in the simulation
solar_mean = 30  # Average solar energy generation in kWh
solar_std = 10  # Standard deviation of solar energy generation in kWh
demand_mean = 40  # Average energy demand in kWh
demand_std = 15  # Standard deviation of energy demand in kWh

# Generate random solar energy and demand profiles for simulation
np.random.seed(0)
solar_profile = np.random.normal(solar_mean, solar_std, num_steps)
demand_profile = np.random.normal(demand_mean, demand_std, num_steps)

# Stochastic Model Predictive Control
def stochastic_mpc(solar_profile, demand_profile, initial_soc, battery_capacity, solar_capacity):
    # Define optimization variables
    battery_soc = cp.Variable(num_steps + 1)
    battery_power = cp.Variable(num_steps)

    # Define the objective (minimize energy costs)
    objective = cp.Minimize(cp.sum(battery_power))

    # Define the constraints
    constraints = [
        battery_soc[0] == initial_soc,
        battery_soc >= 0,
        battery_soc <= battery_capacity,
        battery_soc[1:] == battery_soc[:-1] + battery_power - solar_profile + demand_profile,
        battery_power >= -solar_capacity,
        battery_power <= battery_capacity - solar_profile,
    ]

    # Create and solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    return battery_power.value

# Run the MPC for battery control
battery_power_optimal = stochastic_mpc(solar_profile, demand_profile, initial_battery_soc, battery_capacity, solar_capacity)

# Simulate the energy system using the optimal battery control
battery_soc_simulated = np.zeros(num_steps + 1)
battery_soc_simulated[0] = initial_battery_soc
for t in range(num_steps):
    battery_soc_simulated[t + 1] = battery_soc_simulated[t] + battery_power_optimal[t] - solar_profile[t] + demand_profile[t]

# Print the optimal battery power control and simulated battery SOC
print("Optimal Battery Power Control (kW):", battery_power_optimal)
print("Simulated Battery SOC:", battery_soc_simulated)
