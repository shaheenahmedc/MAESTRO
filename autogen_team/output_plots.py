import matplotlib.pyplot as plt
from datetime import datetime
import os

# Fig 1
# tokens = [526, 594, 526, 594]
# performance = [20, 19.33, 20, 20]
# std_dev = [2, 1.15, 4, 4]
# labels = ["Haiku SAS", "4o SAS", "Haiku SAS + MAS", "4o SAS + MAS"]
# fig_number = "1"
# figure_title = f"50 tasks, All Agents Same"
# # Define markers and colors for each datapoint
# markers = ["s", "D", "o", "^"]  # square, diamond, circle, triangle
# colors = ["#377eb8", "#ff7f00", "#e41a1c", "#4daf4a"]  # blue, orange, red, green


# # Fig 2 data - reordered to put non-MAS entries first, MAS entries last
# tokens = [583, 546, 583, 546]
# performance = [22, 18.67, 25.33, 26]
# std_dev = [7.21, 2.31, 4.16, 2]
# labels = ["Haiku SAS", "4o SAS", "Haiku SAS + 4o MAS", "4o SAS + Haiku MAS"]
# fig_number = "2"
# figure_title = f"50 tasks, Agents Mixed"

# # Define markers and colors for each datapoint (reordered to match)
# markers = ["s", "D", "o", "^"]  # square, diamond, circle, triangle
# colors = ["#377eb8", "#ff7f00", "#e41a1c", "#4daf4a"]  # blue, orange, red, green


# # Fig 3 data
# tokens = [538, 536, 537, 521, 579, 534, 537, 521]
# performance = [24, 28, 18, 26, 28, 20, 18, 22]
# labels = [
#     "4o SAS + Diverse MAS",
#     "Haiku SAS + Diverse MAS",
#     "4o_High_Temp + Diverse SAS",
#     "Haiku_High_Temp + Diverse SAS",
#     "4o SAS",
#     "Haiku SAS",
#     "4o_High_Temp SAS",
#     "Haiku_High_Temp SAS",
# ]
# fig_number = "3"
# figure_title = f"50 tasks, Diverse Agents"

# # Define custom offsets per label (dx, dy), measured in points if textcoords="offset points"
# label_offsets = {
#     labels[0]: (40, 80),
#     labels[1]: (-130, -40),
#     labels[2]: (50, -70),
#     labels[3]: (-20, 70),
#     labels[4]: (10, 10),
#     labels[5]: (-60, -20),
#     labels[6]: (10, 10),
#     labels[7]: (10, -10),
# }
# point_colors = ["red", "red", "red", "red", "blue", "blue", "blue", "blue"]


# Fig 4
# tokens = [532, 532, 600, 600]
# performance = [15, 22, 26, 25]
# labels = ["Haiku SAS + Haiku MAS", "Haiku SAS", "4o SAS + 4o MAS", "4o SAS"]
# fig_number = "4"
# figure_title = f"100 tasks, All Agents Same"
# label_offsets = {
#     labels[0]: (20, 40),
#     labels[1]: (-40, -40),
#     labels[2]: (-80, 80),
#     labels[3]: (-90, -100),
# }
# point_colors = ["red", "blue", "red", "blue"]


# Fig 5 data
# tokens = [556, 565, 600, 532]
# performance = [18, 22, 28, 20]
# labels = ["4o SAS + Haiku MAS", "Haiku SAS + 4o MAS", "4o SAS", "Haiku SAS"]
# fig_number = "5"
# figure_title = f"100 tasks, Agents Mixed"

# # Define custom offsets per label (dx, dy), measured in points if textcoords="offset points"
# label_offsets = {
#     labels[0]: (-40, 40),
#     labels[1]: (-40, -40),
#     labels[2]: (-90, -90),
#     labels[3]: (-80, 80),
# }
# # -- HARD-CODED COLORS --
# # Match each point's color by index in the lists above
# # e.g., "MAS" => red, "SAS" => blue, but you can set them manually here:
# point_colors = ["red", "red", "blue", "blue"]


# # Create plot
# fig, ax = plt.subplots()


# # Plot points with error bars, using different markers and colors
# for i in range(len(tokens)):
#     ax.errorbar(
#         tokens[i],
#         performance[i],
#         yerr=std_dev[i],
#         fmt=markers[i],
#         ms=5,  # markersize
#         mew=2,  # markeredgewidth
#         color=colors[i],
#         ecolor=colors[i],
#         capsize=5,
#         elinewidth=1.5,
#         label=labels[i],
#     )

# ax.set_title(figure_title)

# # Set axis labels
# ax.set_xlabel("Number of Tokens Used ('000s)")
# ax.set_ylabel("Performance (%)")

# # Set axis limits (customize as needed)
# ax.set_xlim(500, 600)
# ax.set_ylim(10, 30)

# # Add legend
# ax.legend(loc="best", frameon=True, framealpha=0.9, shadow=False)

# # Remove grid
# ax.grid(False)
# plt.tight_layout()

# # Save file relative to script's location
# script_dir = os.path.dirname(os.path.abspath(__file__))  # path to this script
# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# filename = f"fig_{fig_number}_{timestamp}.png"
# output_path = os.path.join(script_dir, filename)

# plt.savefig(output_path, dpi=900)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# X-axis values corresponding to increasing tokens (represented as 1, 2, 3 for simplicity)
x = np.array([1, 2, 3])


# Generate clusters around each x-point for 5 samples
def generate_cluster(x_center, y_center, spread=0.04, n=5):
    return (
        x_center + np.random.normal(0, spread, n),
        y_center + np.random.normal(0, spread, n),
    )


# Multi-agent system (MAS) higher performance, steeper increase
mas_y_centers = [0.3, 0.45, 0.6]  # Shifted down from [0.6, 0.75, 0.9]
mas_x, mas_y = [], []
for i, y_c in enumerate(mas_y_centers):
    x_c, y_c = generate_cluster(x[i], y_c)
    mas_x.extend(x_c)
    mas_y.extend(y_c)

# Single-agent system (SAS) lower performance, shallower increase
sas_y_centers = [0.1, 0.15, 0.2]  # Shifted down from [0.4, 0.45, 0.5]
sas_x, sas_y = [], []
for i, y_c in enumerate(sas_y_centers):
    x_c, y_c = generate_cluster(x[i], y_c)
    sas_x.extend(x_c)
    sas_y.extend(y_c)

# Plotting
plt.figure()  # Changed from (10, 6) to make square
plt.scatter(mas_x, mas_y, color="red", label="MAS Experiments")
plt.scatter(sas_x, sas_y, color="blue", label="SAS Experiments")

# Fit lines
mas_fit = np.polyfit(mas_x, mas_y, 1)
sas_fit = np.polyfit(sas_x, sas_y, 1)
x_fit = np.linspace(0.5, 3.5, 100)
plt.plot(x_fit, np.polyval(mas_fit, x_fit), color="red")
plt.plot(x_fit, np.polyval(sas_fit, x_fit), color="blue")

# Axes and labels
plt.xticks(
    x,
    [
        "Reflection Depth ≈ 10/\nRounds ≈ 3/\nTokens ≈ 100k",
        "Reflection Depth ≈ 25/\nRounds ≈ 6/\nTokens ≈ 500k",
        "Reflection Depth ≈ 50/\nRounds ≈ 9/\nTokens ≈ 1000K",
    ],
    fontsize=8,  # Added fontsize parameter to make tick labels smaller
)
plt.xlabel("Tokens Used")
plt.ylabel("Performance (%)")
# Convert y-axis to percentages
from matplotlib.ticker import FuncFormatter

plt.gca().yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: "{:.0f}".format(y * 100))
)
plt.ylim(bottom=0)

plt.title("Performance vs Tokens Used: MAS vs SAS")
plt.legend()
plt.grid(False)  # Changed from True to remove grid
plt.tight_layout()

plt.savefig("fig_6_performance_vs_tokens_used.png", dpi=900)
