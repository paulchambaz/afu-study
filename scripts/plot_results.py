import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(data):
    q1, q3 = np.percentile(data, [25, 75])
    mask = (data >= q1) & (data <= q3)
    iqm = np.mean(data[mask])
    return q1, q3, iqm


def display_results(results_dict, colors_dict, title, N=4):
    interval = 100
    plt.figure(figsize=(10, 6))

    for algo, results in results_dict.items():
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

        plot_indices = slice(0, len(timesteps), N)

        plt.plot(
            timesteps[plot_indices],
            iqm_values[plot_indices],
            color=colors_dict[algo],
            label=f"{algo} IQM",
        )
        plt.fill_between(
            timesteps[plot_indices],
            q1_values[plot_indices],
            q3_values[plot_indices],
            alpha=0.2,
            color=colors_dict[algo],
        )

    plt.xlabel("Training Steps")
    plt.ylabel("Return")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def main() -> None:
    algorithms = ["DDPG", "SAC", "AFU", "AFUPerrin"]
    env_name = "CartPoleContinuousStudy-v0"
    colors_dict = {
        "DDPG": "#D00000",
        "SAC": "#38B000",
        "AFU": "#0096C7",
        "AFUPerrin": "#023E8A",
    }

    # On-policy results
    on_policy_results = {}
    for algo in algorithms:
        with open(f"results/on-policy-{algo}-{env_name}.pk", "rb") as f:
            on_policy_results[algo] = pickle.load(f)
    display_results(
        on_policy_results, colors_dict, "On-Policy Training Progress", N=4
    )

    # Off-policy results
    off_policy_results = {}
    for algo in algorithms:
        with open(f"results/off-policy-{algo}-{env_name}.pk", "rb") as f:
            off_policy_results[algo] = pickle.load(f)
    display_results(
        off_policy_results, colors_dict, "Off-Policy Training Progress", N=1
    )


if __name__ == "__main__":
    main()
