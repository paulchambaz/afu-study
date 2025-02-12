import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(data):
    q1, q3 = np.percentile(data, [25, 75])
    mask = (data >= q1) & (data <= q3)
    iqm = np.mean(data[mask])
    return q1, q3, iqm


def display_results(results):
    interval = 100
    timesteps = np.array(list(results.keys())) * interval

    q1_values = []
    q3_values = []
    iqm_values = []

    for timestep in sorted(results.keys()):
        data = np.array(results[timestep])
        q1, q3, iqm = compute_stats(data)
        q1_values.append(q1)
        q3_values.append(q3)
        iqm_values.append(iqm)

    N = 4
    plot_indices = slice(0, len(timesteps), N)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps[plot_indices], iqm_values[plot_indices], "b-", label="IQM")
    plt.fill_between(
        timesteps[plot_indices],
        q1_values[plot_indices],
        q3_values[plot_indices],
        alpha=0.2,
        color="b",
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Return")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main() -> None:
    # algorithms = ['DDPG', 'SAC', 'AFU']
    algo = "DDPG"
    env_name = "CartPoleContinuousStudy-v0"

    filename = f"results/{algo}-{env_name}.pk"
    with open(filename, "rb") as f:
        ddpg_results = pickle.load(f)

    display_results(ddpg_results)


if __name__ == "__main__":
    main()
