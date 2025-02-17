import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from utilities import create_main_dataframe


def fit_data(x_in, y_in):

    model = LinearRegression()
    model.fit(x_in, y_in)

    y_pred = model.predict(x_in)

    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    res = {"y_pred": y_pred,
         "m_inter": model.intercept_[0],
         "m_coef": model.coef_[0][0],
         "MAE": mae,
         "MSE": mse,
         "RMSE": rmse,
         "R2": r2
         }

    return res


task2_path = ".\\task_2_sensor_calibration"

main_frame = create_main_dataframe(task2_path, measurement_type="calibration")

main_frame["OxygenC"] = main_frame["MFC2"] / (main_frame["MFC1"] + main_frame["MFC2"]) * 20.9

current_data = main_frame[main_frame["DataType"] == " Current"].copy()

current_data["Current"] = current_data["Current"] * 1e3




sensors = current_data["ChNum"].unique()
categories = current_data["ConcNum"].unique()
num_categories = len(categories)
colormap = plt.colormaps['tab10'].colors
category_colors = {category: colormap[j] for j, category in enumerate(categories)}




one_sensor_one_m = current_data[
    (current_data["ChNum"] == "F22")&
    (current_data["MeasureNum"] == 0)&
    (current_data["SweepNum"] == 0)&
    (current_data["ConcNum"] == 0)
].copy()

t_1 = 0.09
t_2 = 0.11
t_start = 0
t_mid = 0.12
t_end = one_sensor_one_m["Time"].iloc[-1]

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex=True)

ax[0].axvspan(t_start, t_mid, color='red', alpha=0.05)
ax[0].axvspan(t_mid, t_end, color='blue', alpha=0.05)
ax[1].axvspan(t_start, t_mid, color='red', alpha=0.05)
ax[1].axvspan(t_mid, t_end, color='blue', alpha=0.05)
ax[0].axvspan(t_1, t_2, color='green', alpha=0.25)


ax[0].plot(one_sensor_one_m["Time"], one_sensor_one_m["Current"], linewidth=2, c="black")
ax[1].plot(one_sensor_one_m["Time"], one_sensor_one_m["Voltage"], linewidth=2, c="black")

ax[1].set_xlabel("Time [s]")
ax[0].set_ylabel("Current [nA]")
ax[1].set_ylabel("Voltage [V]")

ax[1].text(0, -0.5, "Negative Polarization", color="red")
ax[1].text(0.13, -0.5, "Positive Polarization", color="blue")

ax[0].grid(True)
ax[1].grid(True)

fig.tight_layout(h_pad=0.1)






fig0, ax0 = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))
for i in range(4):
    one_sensor = current_data[(current_data["ChNum"] == sensors[i])].copy()
    one_sensor_groups = one_sensor.groupby(["SweepNum", "MeasureNum", "ConcNum"])
    osm = one_sensor.groupby(["ConcNum"])["OxygenC"].agg("mean").reset_index()

    selected_concs = set()
    for _, group in one_sensor_groups:
        conc_num = group["ConcNum"].unique()[0]
        col = category_colors[conc_num]
        conc_lbl = osm[osm["ConcNum"]==conc_num]["OxygenC"].values[0]
        label = f"O$_2$: {round(conc_lbl, 2)} [vol%]"

        if conc_num not in selected_concs:
            ax0[0, i].plot(group["Time"], group["Current"], c=col, label=label)
            ax0[1, i].plot(group["Time"], group["Voltage"], c=col, label=label)
        else:
            ax0[0, i].plot(group["Time"], group["Current"], c=col)
            ax0[1, i].plot(group["Time"], group["Voltage"], c=col)

        selected_concs.add(conc_num)
        ax0[1, i].set_xlabel("Time [s]")

    ax0[0, i].set_title(f"Channel {sensors[i]}")

    ax0[0, i].legend(title="Mean Concentration", loc="lower right")
    ax0[1, i].legend(title="Mean Concentration", loc="lower right")

ax0[0, 0].set_ylabel("Current [nA]")
ax0[1, 0].set_ylabel("Voltage [V]")

fig0.tight_layout()




t1 = 0.09
t2 = 0.11

current_data = current_data[(current_data["Time"] > t1) & (current_data["Time"] < t2)]




fig1, ax1 = plt.subplots(nrows=3, ncols=4, figsize=(18, 8))
for i in range(4):
    one_sensor = current_data[(current_data["ChNum"] == sensors[i])].copy()
    one_sensor_mean = one_sensor.groupby(["SweepNum", "MeasureNum", "ConcNum"])[["Current", "OxygenC", "t_sht", "h_sht"]].mean().reset_index()
    one_sensor_mean["Color"] = one_sensor_mean["ConcNum"].map(category_colors)

    ax1[0, i].scatter(one_sensor_mean["OxygenC"], one_sensor_mean["Current"],
                   c=one_sensor_mean["Color"], s=50, edgecolors="gray", alpha=0.7)
    ax1[0, i].set_xlabel("O$_2$ [vol%]")

    ax1[1, i].scatter(one_sensor_mean["h_sht"], one_sensor_mean["Current"],
                   c=one_sensor_mean["Color"], s=50, edgecolors="gray", alpha=0.7)
    ax1[1, i].set_xlabel("Relative Humidity [%]")

    ax1[2, i].scatter(one_sensor_mean["t_sht"], one_sensor_mean["Current"],
                   c=one_sensor_mean["Color"], s=50, edgecolors="gray", alpha=0.7)
    ax1[2, i].set_xlabel("Temperature [C]")

    ax1[0, i].set_title(f"Channel {sensors[i]}")

ax1[0, 0].set_ylabel("Current [nA]")
ax1[1, 0].set_ylabel("Current [nA]")
ax1[2, 0].set_ylabel("Current [nA]")
fig1.tight_layout()






calib_data = []

fig2, ax2 = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))
for i in range(4):
    one_sensor = current_data[(current_data["ChNum"] == sensors[i])].copy()
    one_sensor_mean = one_sensor.groupby(["SweepNum", "MeasureNum", "ConcNum"])[["Current", "OxygenC", "t_sht", "h_sht"]].mean().reset_index()
    one_sensor_mean["Color"] = one_sensor_mean["ConcNum"].map(category_colors)

    x = one_sensor_mean["OxygenC"].values.reshape(-1, 1)
    y = one_sensor_mean["Current"].values.reshape(-1, 1)
    cd = fit_data(x, y)
    y_ = cd["y_pred"]
    cd.pop("y_pred", None)
    calib_data.append(cd)

    ax2[i].scatter(one_sensor_mean["OxygenC"], one_sensor_mean["Current"],
                   c=one_sensor_mean["Color"], s=50, edgecolors="gray", alpha=0.7)
    ax2[i].set_xlabel("O$_2$ [vol%]")

    ax2[i].plot(x, y_ , c="black", linestyle=":")

    ax2[i].set_title(f"Channel {sensors[i]}")

ax2[0].set_ylabel("Current [nA]")

fig2.tight_layout()

calib_data = pd.DataFrame(calib_data)

print(calib_data)


plt.show()