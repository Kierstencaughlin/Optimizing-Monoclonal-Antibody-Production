import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor

from src.classes import StandardScalerVAR
from src.functions import generate_x_lag_cur_var, progress_bar

# Number of forecast simulations
n_simulations = 1000

# Max order to evaluate
p_max = 3

# Fraction of simulated forecasts used to calculate prediction intervals
prediction_interval = 0.95
alpha = 1 - prediction_interval

# Create a random number generator
rng = np.random.default_rng(0)

# Import the initial conditions of each batch
df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))

# Import the 30 selected batch datasets
batches = []
for filepath in sorted(Path("datasets").glob("batch_*.csv")):
    batch_id = int(filepath.stem.split("_")[1])
    df_batch = pd.read_csv(filepath)
    df_batch.insert(0, "Batch ID", batch_id)
    batches.append(df_batch)

df = pd.concat(batches, ignore_index=True)
df = df.sort_values(["Batch ID", "Time [h]"]).reset_index(drop=True)

# Extract the unique batch IDs
batch_ids = np.unique(df.loc[:, "Batch ID"])

# Select all process variables to include in the VAR model
features = df.columns[2:]

# Select the variable that will be plotted and evaluated
feature_plot = "mAb [mg/L]"
idx_fp = np.argmax(features == feature_plot)

# Extract units for plotting and reporting
units = feature_plot.split(" [")[-1][:-1]

n_iters = p_max * batch_ids.shape[0] * n_simulations
c_iter = 0
records = []

# Evaluate VAR models with different numbers of lagged values
for p in range(1, p_max + 1):
    fig, axes = plt.subplots(6, 5, figsize=(16, 18), dpi=200, layout="constrained")

    y_min = 0
    y_max = np.max(df.loc[:, feature_plot])
    rmse_forecasts = []

    # Leave-one-batch-out cross-validation
    for k, batch_id_test in enumerate(batch_ids):

        # Training batches contain all batches except the test batch
        batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

        # Generate lagged values used to train VAR(p)
        x_lag_train, x_cur_train, _, _ = generate_x_lag_cur_var(
            p=p, batch_ids=batch_ids_train, df=df, features=features
        )

        # Create and train the model
        model = make_pipeline(StandardScalerVAR(p=p), LinearRegression())
        model.fit(x_lag_train, x_cur_train)

        # Calculate residual errors on the training data
        x_cur_train_pred = model.predict(x_lag_train)
        x_cur_train_res = x_cur_train_pred - x_cur_train

        # Extract the initial conditions for the test batch
        idx_ic = (df_ic.loc[:, "Batch ID"] == batch_id_test).idxmax()
        initial_glucose = df_ic.at[idx_ic, "Glucose [mM]"]
        initial_glutamine = df_ic.at[idx_ic, "Glutamine [mM]"]
        initial_vcd = df_ic.at[idx_ic, "VCD [cells/mL]"]

        # Initial state: glucose, glutamine, lactate, ammonia, VCD, TCD, mAb
        # Lactate, ammonia, and mAb start at zero; initial TCD equals VCD.
        x_initial = np.array([
            initial_glucose, initial_glutamine, 0, 0,
            initial_vcd, initial_vcd, 0
        ])

        # One subplot for each selected batch
        ax = axes.flat[k]

        # Extract the data for the test batch
        mask = df.loc[:, "Batch ID"] == batch_id_test
        t_test = df.loc[mask, "Time [h]"].to_numpy()
        x_test = df.loc[mask, features].to_numpy()

        # Perform multiple forecast simulations to estimate uncertainty
        x_forecasts = []
        for simulation_id in range(n_simulations):
            x_forecast = [x_initial]

            # Generate forecasts recursively
            while len(x_forecast) < t_test.shape[0]:
                x_lag = x_forecast[::-1][:p].copy()

                # Repeat the latest state if fewer than p lagged vectors exist
                while len(x_lag) < p:
                    x_lag.append(x_lag[-1])

                # Concatenate lagged vectors into one row
                x_lag = np.concatenate(x_lag, axis=0)
                x_lag = np.expand_dims(x_lag, axis=0)

                # Sample residual noise separately for each process variable
                white_noise = []
                for j in range(x_cur_train_res.shape[1]):
                    white_noise.append(rng.choice(x_cur_train_res[:, j]))
                white_noise = np.array([white_noise])

                # Predict next process state and add noise
                x_cur = model.predict(x_lag) + white_noise
                x_cur = np.clip(x_cur[0], a_min=0, a_max=None)
                x_forecast.append(x_cur)

            x_forecasts.append(np.array(x_forecast))

            c_iter += 1
            progress_bar(c_iter, n_iters)

        x_forecasts = np.array(x_forecasts)

        # Calculate median forecast and 95% prediction intervals
        x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
        x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
        x_forecast_pi_upper = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

        # Calculate forecast RMSE for mAb concentration
        rmse_forecast = root_mean_squared_error(
            x_test[:, idx_fp], x_forecast_median[:, idx_fp]
        )
        rmse_forecasts.append(rmse_forecast)

        message = f"RMSE: {rmse_forecast:.2f} {units}\n"
        message += f"GLC: {initial_glucose:.0f} mM\n"
        message += f"GLN: {initial_glutamine:.0f} mM\n"
        message += f"VCD: {initial_vcd:.0f} cells/mL"

        # Plot observed values, forecast median, and prediction interval
        ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=10,
                   color="tab:orange", edgecolors="black", alpha=0.7)
        ax.plot(t_test, x_forecast_median[:, idx_fp],
                label=f"VAR({p})", color="black")
        ax.fill_between(
            t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_upper[:, idx_fp],
            label=f"{prediction_interval:.0%} PI", color="tab:blue", alpha=0.3
        )
        ax.text(0.02, 0.96, message, transform=ax.transAxes,
                va="top", ha="left", fontsize=7, fontname="monospace")
        ax.set_title(f"Batch {batch_id_test:04d}", fontsize=9)

        # Track maximum value for consistent y-axis limits
        y_max = max(y_max, np.max(x_forecast_pi_upper[:, idx_fp]))

    # Apply consistent axis limits and formatting to all subplots
    y_buffer = 0.05 * (y_max - y_min)
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            ax = axes[i, j]
            ax.set_ylim(bottom=y_min - y_buffer, top=y_max + y_buffer)

            if i == axes.shape[0] - 1:
                ax.set_xlabel("Time [h]")
                ax.xaxis.set_major_locator(MultipleLocator(96))
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(feature_plot)
            else:
                ax.set_yticks([])

            if i == 0 and j == 0:
                ax.legend(loc="upper right", fontsize=7)

    # Save figure and summarize forecast error
    fig.savefig(os.path.join("figures", f"var_p{p}.png"))
    plt.close(fig)

    rmse_mean = np.mean(rmse_forecasts)
    rmse_sd = np.std(rmse_forecasts)
    records.append((p, n_simulations, rmse_mean, rmse_sd))
    print(f"VAR({p}) | RMSE: {rmse_mean:.2f} +/- {rmse_sd:.2f} {units}")

# Export the model comparison table
cols_export = ["p", "N simulations", f"RMSE (mean) [{units}]", f"RMSE (SD) [{units}]"]
df_export = pd.DataFrame.from_records(records, columns=cols_export)
df_export.to_csv(os.path.join("tables", "var_results.csv"), index=False, float_format="%.2f")

# ============================================================================
# Non-linear Vector Autoregression - mAb Production
# ============================================================================

# Parameters for NVAR
cv_splits = 3
poly_degree_max=3
knn_k_max=5
model_types = ["poly", "knn"]
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(2, poly_degree_max+1)])
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(2, knn_k_max + 1)],kneighborsregressor__weights=["uniform"]
)
# Tune the model using mAb RMSE rather than the average RMSE across
# process variables with very different units (e.g., VCD and mAb).
def score_mab(model, x, y):
    y_pred = model.predict(x)
    return -root_mean_squared_error(y[:, idx_fp], y_pred[:, idx_fp])

rng = np.random.default_rng(0)
n_iters = len(model_types) * p_max * batch_ids.shape[0] * n_simulations
c_iter = 0
records = []

for model_type in model_types:
    for p in range(1, p_max + 1):
        fig, axes = plt.subplots(6, 5, figsize=(16, 18), dpi=200, layout="constrained")

        y_min = 0
        y_max = np.max(df.loc[:, feature_plot])
        rmse_forecasts = []

        # Leave-one-batch-out cross-validation
        for k, batch_id_test in enumerate(batch_ids):

            # Train on all batches except the test batch
            batch_ids_train = np.array([i for i in batch_ids if i != batch_id_test])

            # Generate lagged values and batch groups
            x_lag_train, x_cur_train, _, b_cur_train = generate_x_lag_cur_var(
                p=p, batch_ids=batch_ids_train, df=df, features=features
            )

            # Limits for each process variable separately
            a_min = np.min(x_cur_train,axis=0)
            a_max = np.max(x_cur_train,axis=0)

            # Create the nonlinear model
            if model_type == "poly":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    PolynomialFeatures(include_bias=False),
                    LinearRegression()
                )
                param_grid = param_grid_poly
            elif model_type == "knn":
                model = make_pipeline(
                    StandardScalerVAR(p=p),
                    KNeighborsRegressor()
                )
                param_grid = param_grid_knn
            else:
                model = None
                param_grid = {}

            # Tune using folds that keep entire batches together
            gkf = GroupKFold(n_splits=cv_splits)
            model_gscv = GridSearchCV(
                model, param_grid=param_grid, cv=gkf,
                scoring=score_mab, return_train_score=True
            )
            model_gscv.fit(x_lag_train, x_cur_train, groups=b_cur_train)

            # Calculate training residuals for white-noise simulations
            x_cur_train_pred = model_gscv.predict(x_lag_train)
            x_cur_train_res = x_cur_train_pred - x_cur_train

            # Initial conditions for the excluded batch
            idx_ic = (df_ic.loc[:, "Batch ID"] == batch_id_test).idxmax()
            initial_glucose = df_ic.at[idx_ic, "Glucose [mM]"]
            initial_glutamine = df_ic.at[idx_ic, "Glutamine [mM]"]
            initial_vcd = df_ic.at[idx_ic, "VCD [cells/mL]"]

            # Order: glucose, glutamine, lactate, ammonia, VCD, TCD, mAb
            x_initial = np.array([
                initial_glucose, initial_glutamine, 0, 0,
                initial_vcd, initial_vcd, 0
            ])

            # Get the test measurements
            mask = df.loc[:, "Batch ID"] == batch_id_test
            t_test = df.loc[mask, "Time [h]"].to_numpy()
            x_test = df.loc[mask, features].to_numpy()
            ax = axes.flat[k]

            # Forecast with residual noise
            x_forecasts = []
            for simulation_id in range(n_simulations):
                x_forecast = [x_initial]

                while len(x_forecast) < t_test.shape[0]:
                    x_lag = x_forecast[::-1][:p].copy()

                    while len(x_lag) < p:
                        x_lag.append(x_lag[-1])

                    x_lag = np.concatenate(x_lag, axis=0)
                    x_lag = np.expand_dims(x_lag, axis=0)

                    white_noise = []
                    for j in range(x_cur_train_res.shape[1]):
                        white_noise.append(rng.choice(x_cur_train_res[:, j]))
                    white_noise = np.array([white_noise])

                    x_cur_baseline = model_gscv.predict(x_lag)
                    x_cur_baseline = np.clip(x_cur_baseline, a_min=a_min, a_max=a_max)
                    x_cur = x_cur_baseline + white_noise
                    x_cur = np.clip(x_cur[0], a_min=0, a_max=None)
                    x_forecast.append(x_cur)


                x_forecast = np.array(x_forecast)
                x_forecasts.append(np.array(x_forecast))
                c_iter += 1
                progress_bar(c_iter, n_iters)


            # Median forecast and 95% prediction interval
            x_forecast_pi_lower = np.quantile(x_forecasts, q=alpha / 2, axis=0)
            x_forecast_median = np.quantile(x_forecasts, q=0.5, axis=0)
            x_forecast_pi_upper = np.quantile(x_forecasts, q=1 - alpha / 2, axis=0)

            # Evaluate the forecast of mAb concentration
            rmse_forecast = root_mean_squared_error(x_test[:, idx_fp], x_forecast_median[:, idx_fp])
            rmse_forecasts.append(rmse_forecast)

            message = f"RMSE: {rmse_forecast:.2f} {units}\n"
            message += f"GLC: {initial_glucose:.0f} mM\n"
            message += f"GLN: {initial_glutamine:.0f} mM\n"
            message += f"VCD: {initial_vcd:.0f} cells/mL\n"


            # Extract the best hyperparameters
            if model_type == "poly":
                best_degree = model_gscv.best_params_["polynomialfeatures__degree"]
                message += f"Degree: {best_degree}"

            elif model_type == "knn":
                best_knn = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
                best_weight = model_gscv.best_params_["kneighborsregressor__weights"]
                message += f"k: {best_knn}, {best_weight}"

            # Plot observations, median forecast, and prediction interval
            ax.scatter(t_test, x_test[:, idx_fp], label="Test", s=10,
                       color="tab:orange", edgecolors="black", alpha=0.7)
            ax.plot(t_test, x_forecast_median[:, idx_fp],
                    label=f"NVAR({p}) {model_type}", color="black")
            ax.fill_between(
                t_test, x_forecast_pi_lower[:, idx_fp], x_forecast_pi_upper[:, idx_fp],
                label=f"{prediction_interval:.0%} PI", color="tab:blue", alpha=0.3
            )
            ax.text(0.02, 0.96, message, transform=ax.transAxes,
                    va="top", ha="left", fontsize=7, fontname="monospace")
            ax.set_title(f"Batch {batch_id_test:04d}", fontsize=9)

            y_max = max(y_max, np.max(x_forecast_pi_upper[:, idx_fp]))

        # Use common y-axis limits for all 30 batches
        y_buffer = 0.05 * (y_max - y_min)
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                ax = axes[i, j]
                ax.set_ylim(bottom=y_min - y_buffer, top=y_max + y_buffer)

                if i == axes.shape[0] - 1:
                    ax.set_xlabel("Time [h]")
                    ax.xaxis.set_major_locator(MultipleLocator(96))
                else:
                    ax.set_xticks([])

                if j == 0:
                    ax.set_ylabel(feature_plot)
                else:
                    ax.set_yticks([])

                if i == 0 and j == 0:
                    ax.legend(loc="upper right", fontsize=7)

        fig.savefig(os.path.join("figures", f"nvar_{model_type}_p{p}.png"))
        plt.close(fig)

        rmse_mean = np.mean(rmse_forecasts)
        rmse_sd = np.std(rmse_forecasts)
        records.append((model_type, p, n_simulations, rmse_mean, rmse_sd))
        print(f"NVAR({p}) {model_type} | RMSE: {rmse_mean:.2f} +/- {rmse_sd:.2f} {units}")

# Export NVAR results separately for manual comparison with VAR
cols_export = ["Model", "p", "N simulations", f"RMSE (mean) [{units}]", f"RMSE (SD) [{units}]"]
df_export = pd.DataFrame.from_records(records, columns=cols_export)
df_export.to_csv(os.path.join("tables", "nvar_results.csv"), index=False, float_format="%.2f")

# ============================================================================
# Space-Time Profit Optimization - Polynomial NVAR(3)
# ============================================================================

# Train the selected model using all 30 batches
p = 3
x_lag_train, x_cur_train, _, _ = generate_x_lag_cur_var(p=p, batch_ids=batch_ids, df=df, features=features)

model = make_pipeline(StandardScalerVAR(p=p),PolynomialFeatures(degree=2, include_bias=False),LinearRegression())
model.fit(x_lag_train, x_cur_train)

# Use training residuals to estimate uncertainty after choosing the optimum
residuals = x_cur_train - model.predict(x_lag_train)
x_max = np.max(x_cur_train, axis=0)

# All unique combinations of initial glucose, glutamine and VCD
ic_features = ["Glucose [mM]", "Glutamine [mM]", "VCD [cells/mL]"]
df_conditions = df_ic[ic_features].drop_duplicates().reset_index(drop=True)
glucose = df_conditions["Glucose [mM]"].to_numpy()
glutamine = df_conditions["Glutamine [mM]"].to_numpy()
vcd = df_conditions["VCD [cells/mL]"].to_numpy()

# Initial state: glucose, glutamine, lactate, ammonia, VCD, TCD, mAb
x_initial = np.column_stack((glucose, glutamine, np.zeros(len(vcd)),np.zeros(len(vcd)), vcd, vcd, np.zeros(len(vcd))))

# Forecast all 343 conditions at the experimental sampling times
hours = df.loc[df["Batch ID"] == batch_ids[0], "Time [h]"].to_numpy()
days = hours / 24
x_forecast = [x_initial]

while len(x_forecast) < len(hours):
    x_lag = x_forecast[::-1][:p].copy()
    while len(x_lag) < p:
        x_lag.append(x_lag[-1])
    x_lag = np.concatenate(x_lag, axis=1)
    x_cur = np.clip(model.predict(x_lag), a_min=0, a_max=x_max)
    x_forecast.append(x_cur)

mab = np.array(x_forecast)[:, :, idx_fp].T  # condition x time
cost = 10 * glucose + 20 * glutamine + 0.0005 * vcd
profit = (1000 * mab - cost[:, None]) / (days[None, :] + 1)
profit[:, 0] = -np.inf

# Find the most profitable harvest time for each initial condition
idx_harvest = np.argmax(profit, axis=1)
idx_conditions = np.arange(len(df_conditions))
df_results = df_conditions.copy()
df_results["Harvest time [d]"] = days[idx_harvest]
df_results["Culture cost [$/L]"] = cost
df_results["Space-time profit [$/L/d]"] = profit[idx_conditions, idx_harvest]
df_results["Space-time yield [mg/L/d]"] = mab[idx_conditions, idx_harvest] / (days[idx_harvest] + 1)
df_results.to_csv(os.path.join("tables", "optimization_grid.csv"), index=False, float_format="%.2f")

# Select the highest space-time profit across all conditions and harvest times
idx_best, idx_time = np.unravel_index(np.argmax(profit), profit.shape)
best_glucose = glucose[idx_best]
best_glutamine = glutamine[idx_best]
best_vcd = vcd[idx_best]
best_day = days[idx_time]
best_cost = cost[idx_best]
best_mab = mab[idx_best, idx_time]

# Run 1000 simulations at the selected conditions for a 95% prediction interval
rng = np.random.default_rng(0)
x_simulations = [np.tile(x_initial[idx_best], (n_simulations, 1))]

for _ in range(idx_time):
    x_lag = x_simulations[::-1][:p].copy()
    while len(x_lag) < p:
        x_lag.append(x_lag[-1])
    x_lag = np.concatenate(x_lag, axis=1)

    white_noise = np.column_stack([
        rng.choice(residuals[:, j], size=n_simulations)
        for j in range(len(features))
    ])
    x_cur = np.clip(model.predict(x_lag), a_min=0, a_max=x_max)
    x_cur = np.clip(x_cur + white_noise, a_min=0, a_max=None)
    x_simulations.append(x_cur)

mab_lower, mab_upper = np.quantile(
    x_simulations[-1][:, idx_fp], [0.025, 0.975]
)

# Value, profit and yield are linear functions of mAb at the fixed harvest time
value = 1000 * best_mab
value_lower, value_upper = 1000 * mab_lower, 1000 * mab_upper
space_time_profit = (value - best_cost) / (best_day + 1)
profit_lower = (value_lower - best_cost) / (best_day + 1)
profit_upper = (value_upper - best_cost) / (best_day + 1)
space_time_yield = best_mab / (best_day + 1)
yield_lower, yield_upper = mab_lower / (best_day + 1), mab_upper / (best_day + 1)

# Export the results required for the README
summary = {
    "Initial glucose [mM]": best_glucose,
    "Initial glutamine [mM]": best_glutamine,
    "Initial VCD [cells/mL]": best_vcd,
    "Harvest time [d]": best_day,
    "Cell culture cost [$/L]": best_cost,
    "Cell culture value [$/L]": value,
    "Value 95% PI lower [$/L]": value_lower,
    "Value 95% PI upper [$/L]": value_upper,
    "Space-time profit [$/L/d]": space_time_profit,
    "Profit 95% PI lower [$/L/d]": profit_lower,
    "Profit 95% PI upper [$/L/d]": profit_upper,
    "Space-time yield [mg/L/d]": space_time_yield,
    "Yield 95% PI lower [mg/L/d]": yield_lower,
    "Yield 95% PI upper [mg/L/d]": yield_upper
}
pd.DataFrame([summary]).to_csv(os.path.join("tables", "optimization_summary.csv"), index=False, float_format="%.2f")

print("\nRecommended operating conditions: Polynomial NVAR(3)")
print(f"Glucose: {best_glucose:.0f} mM | Glutamine: {best_glutamine:.0f} mM | VCD: {best_vcd:.0f} cells/mL")
print(f"Harvest: {best_day:.2f} d | Cost: ${best_cost:.2f}/L")
print(f"Value: ${value:.2f}/L (95% PI: ${value_lower:.2f} to ${value_upper:.2f})")
print(f"Profit: ${space_time_profit:.2f}/L/d (95% PI: ${profit_lower:.2f} to ${profit_upper:.2f})")
print(f"Yield: {space_time_yield:.3f} mg/L/d (95% PI: {yield_lower:.3f} to {yield_upper:.3f})")

# Plot how each initial condition affects profit and yield
fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=200, layout="constrained")
for i, feature in enumerate(ic_features):
    mask = np.ones(len(df_results), dtype=bool)
    for j, other_feature in enumerate(ic_features):
        if j != i:
            mask &= df_results[other_feature] == df_results.loc[idx_best, other_feature]
    df_plot = df_results.loc[mask].sort_values(feature)
    axes[0, i].plot(df_plot[feature], df_plot["Space-time profit [$/L/d]"], marker="o")
    axes[1, i].plot(df_plot[feature], df_plot["Space-time yield [mg/L/d]"], marker="o")
    axes[0, i].set_xlabel(feature)
    axes[1, i].set_xlabel(feature)
    axes[0, i].axvline(df_results.loc[idx_best, feature], linestyle="--", color="tab:red")
    axes[1, i].axvline(df_results.loc[idx_best, feature], linestyle="--", color="tab:red")
    if i == 0:
        axes[0, i].set_ylabel("Space-time profit [$/L/d]")
        axes[1, i].set_ylabel("Space-time yield [mg/L/d]")
fig.savefig(os.path.join("figures", "initial_condition_effects.png"))
plt.close(fig)
