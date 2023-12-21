import matplotlib.pyplot as plt
import scienceplots

phi = (1 + 5**0.5) / 2

plt.style.use(["science", "grid", "no-latex"])
plt.rcParams.update(
    {
        "figure.figsize": [6 * phi, 6],
        "figure.dpi": 300,
        # "legend.fontsize": "small",
    }
)
