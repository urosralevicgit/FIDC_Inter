import matplotlib.pyplot as plt
import numpy as np
from utilities import create_main_dataframe
from scipy.stats import normaltest
from scipy.signal import welch


def normal_d_test(rnd_var):
    _, p_val = normaltest(rnd_var)
    return p_val




task1_path = ".\\task_1_noise_analysis"

main_frame = create_main_dataframe(task1_path)

main_frame.sort_values(by=["ChNum", "MeasureNum"])

main_frame["Current"] = main_frame["Current"] * 1e6




channel_number = 3
measurement_number = 0

single_measurement = main_frame[
    (main_frame["ChNum"] == 2) &
    (main_frame["MeasureNum"] == 2)
].copy().reset_index()

fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(7, 3))
x = single_measurement["Time"]
y = single_measurement["Current"]
ax[0].plot(x, y, linewidth=1)
ax[0].set_ylabel("Current [pA]")
ax[0].set_title(f"Channel {channel_number}, Measurement: {measurement_number}", fontsize=10)
ax[0].set_xlabel("Time [s]")

ax[1].plot(x.iloc[50:], y.iloc[50:], linewidth=1)
ax[1].set_ylabel("Current [pA]")
ax[1].set_xlabel("Time [s]")

fig.tight_layout()



cut_idx = 50
main_frame = main_frame.drop(main_frame.index[:cut_idx]).reset_index(drop=True)
channel_groups = main_frame.groupby(["ChNum"])
channel_measurement_groups = main_frame.groupby(["ChNum", "MeasureNum"])




fig0, ax0 = plt.subplots(nrows=8, ncols=1, figsize=(12, 8))
for row, (_, group) in enumerate(channel_groups):
    ch_num = group["ChNum"].unique()
    median = group["Current"].median()
    std = group["Current"].std()
    for i in range(group["MeasureNum"].max()):
        single_measurement = group[group["MeasureNum"] == i]
        x = single_measurement["Time"]
        y = single_measurement["Current"]
        ax0[row].plot(x, y, linewidth=0.5)
        ax0[row].set_ylabel("Current [pA]")
    ax0[row].set_title(f"Channel {ch_num}, Median: {round(median)}, StandarDev: {round(std)}", fontsize=10)
ax0[-1].set_xlabel("Time [s]")
fig0.tight_layout()


fig1, ax1 = plt.subplots(nrows=1, ncols=8, figsize=(16, 2))
for row, (_, group) in enumerate(channel_groups):
    ch_num = group["ChNum"].unique()
    median = group["Current"].median()
    std = group["Current"].std()
    y = group["Current"]
    ax1[row].hist(y, bins=30, edgecolor="gray", linewidth=0.5, density=True)

    ax1[row].set_title(f"Channel {ch_num}\n Median: {median:.2f}\nStd Dev: {std:.2f}", fontsize=10)
    ax1[row].set_xlabel("Current [pA]")

ax1[0].set_ylabel("Density")
fig1.tight_layout()






ch_psd_channels = dict()
frequencies = np.array([])
for _, chg in channel_groups:
    channel_im_groups = chg.groupby("MeasureNum")
    ch_psd_arr=[]
    ch_num = chg["ChNum"].iloc[0]
    for _, chimg in channel_im_groups:
        dt = chimg["Time"].diff().mean()
        f = 1 / dt
        frequencies, ch_psd = welch(chimg["Current"], fs=f, nperseg=256)

        # print(normaltest(chimg["Current"]))

        ch_psd_arr.append(ch_psd)
    ch_psd_channels[ch_num] = np.mean(ch_psd_arr, axis=0)



fig2, ax2 = plt.subplots(nrows=1, ncols=8, figsize=(16, 2))
for key, val in ch_psd_channels.items():
    psd = val
    ax2[key].semilogx(frequencies, 10 * np.log10(psd), label=f'Channel {key}')
    ax2[key].set_title(f"Channel {key}", fontsize=10)
    ax2[key].set_xlabel("Frequency [Hz]")
    ax2[key].grid(which='both')

ax2[0].set_ylabel("PSD [dB/Hz]")
fig2.tight_layout()






stats = channel_measurement_groups["Current"].agg(
    median='median',
    std='std',
    normal_p_val=normal_d_test,
    mean='mean'
).reset_index()

print(stats)
num_norm = (stats["normal_p_val"] > 0.05).sum()
print(f"Number of channels which passed D'Agostino Pearson normal test{num_norm}")

categories = stats["ChNum"].unique()
num_categories = len(categories)
colormap = plt.colormaps['tab10']
category_colors = {category: colormap(i / num_categories) for i, category in enumerate(categories)}
stats["Color"] = stats["ChNum"].map(category_colors)

fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
for category in categories:
    category_data = stats[stats["ChNum"] == category]
    ax3.scatter(category_data["median"], category_data["std"],
                c=category_data["Color"], label=f"ChNum {category}", s=50, edgecolors="gray", alpha=0.7)
ax3.legend(title="Channel Number")
ax3.set_xlabel("Median Current [pA]")
ax3.set_ylabel("Current Standard Deviation [pA]")
fig3.tight_layout()

plt.show()

