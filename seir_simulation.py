import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

# generate SEIR datasets
for seed in range(20, 100):
    for alpha in [0.15]:
        # parameter set for beta, gamma, sigma
        for params in [(6.0, 0.6, 1.8)]:
            np.random.seed(seed)

            b, g, sigma, alpha_value, N = params[0], params[1], params[2], alpha, 1.0


            # encode ODEs for solve_ivp data-generation processes
            def SEIR(t, y):
                # unpack y
                S, E, I, R = tuple(y)

                # SEIR model equations
                dSdt = -b * S * I / N
                dEdt = b * S * I / N - sigma * E
                dIdt = sigma * E - g * I
                dRdt = g * I

                # return only the derivatives
                return np.array([dSdt, dEdt, dIdt, dRdt])


            # initial conditions
            SEIR_init = np.array([0.9899, 0.0050, 0.0050, 0.0001])
            t_start, t_end = 0.0, 20.0
            t_steps = np.linspace(start=t_start, stop=t_end, num=20001)

            # solve the system
            X = solve_ivp(fun=SEIR, t_span=(t_start, t_end), y0=SEIR_init,
                          t_eval=t_steps, atol=1e-10, rtol=1e-10).y.T

            # Compute noise levels (on log scale)
            X_log = np.log(X)

            # Add noise to the data (on log scale)
            X_noised_log = X_log.copy()
            X_noised_log += np.random.normal(loc=0.0, scale=alpha_value, size=X_log.shape)
            X_noised = np.exp(X_noised_log)

            # save time, X_noised, and true values
            data = np.hstack([t_steps.reshape(-1, 1), X_noised, X])
            cols = ["t", "S_obs", "E_obs", "I_obs", "R_obs",
                    "S_true", "E_true", "I_true", "R_true"]

            df = pd.DataFrame(data=data, columns=cols)
            filename = f"tfpigp/data/logSEIR_beta={b}_gamma={g}_sigma={sigma}_alpha={alpha_value}_seed={seed}.csv"
            df.to_csv(filename, index=False)




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
beta, gamma, sigma = 6.0, 0.6, 1.8
t_start, t_end = 0.0, 20.0
num_steps = 20001
t_steps = np.linspace(t_start, t_end, num_steps)
initial_conditions_log = np.log([0.9899, 0.005, 0.0001])  # Initial logS, logI, logR

# SEIR function for simulation (log scale)
def f_vec_log(t, logX, thetas):
    logS, logI, logR = logX
    beta, gamma, sigma = thetas

    # Convert log variables back to original scale
    S = np.exp(logS)
    I = np.exp(logI)
    R = np.exp(logR)

    # Compute E implicitly
    E = 1.0 - S - I - R

    # Derivatives in the original scale
    dSdt = -beta * S * I
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I

    # Derivatives in log scale
    dlogSdt = dSdt / S
    dlogIdt = dIdt / I
    dlogRdt = dRdt / R

    return [dlogSdt, dlogIdt, dlogRdt]

# Solve ODE
solution = solve_ivp(
    fun=lambda t, logX: f_vec_log(t, logX, [beta, gamma, sigma]),
    t_span=(t_start, t_end),
    y0=initial_conditions_log,
    t_eval=t_steps,
    atol=1e-10,
    rtol=1e-10
)

# Extract results and convert back from log scale
log_results = solution.y
results = np.exp(log_results)  # Convert back to original scale
S, I, R = results

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_steps, S, label='S (Susceptible)')
plt.plot(t_steps, I, label='I (Infected)')
plt.plot(t_steps, R, label='R (Recovered)')
plt.xlabel('Time')
plt.ylabel('Proportion of Population')
plt.title('SEIR Simulation (Log Scale with E Masked)')
plt.legend()
plt.grid()
plt.show()
