import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(data):
    data = np.array(data)
    q1, q3 = np.percentile(data, [25, 75])
    mask = (data >= q1) & (data <= q3)
    iqm = np.mean(data[mask])
    return q1, q3, iqm


def display_results(results_dict, colors_dict, title, N=4):
    interval = 100
    plt.figure(figsize=(10, 6))

    for algo, results in results_dict.items():
        # Extract rewards data
        rewards_data = results["rewards"]
        timesteps = np.array(list(rewards_data.keys())) * interval

        q1_values = []
        q3_values = []
        iqm_values = []

        for timestep in sorted(rewards_data.keys()):
            data = np.array(rewards_data[timestep])
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
    # plt.show()
    plt.savefig(f"{title}.svg", format="svg")


def plot_histograms(experiment_results, colors_dict, experiment, env):
    for algo, results in experiment_results.items():
        rewards_data = results["rewards"]

        all_timesteps = sorted(list(rewards_data.keys()))

        last_timesteps = all_timesteps[-int(len(all_timesteps) * 0.01) :]
        # last_timesteps = all_timesteps[-1:]

        all_rewards = []
        for timestep in last_timesteps:
            all_rewards.extend(rewards_data[timestep])

        plt.figure(figsize=(10, 6))
        plt.hist(all_rewards, bins=50, color=colors_dict[algo], alpha=0.7)
        plt.title(
            f"{env} {algo} Reward Distribution - Last 1% of {experiment} Training"
        )
        # plt.title(f"{algo} Reward Distribution - Last episode {experiment} Training")
        plt.xlabel("Return")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()

        # svg_filename = f"histogram_{env}_{algo}_{experiment}.svg"
        # plt.savefig(svg_filename, format="svg")
        # print(f"Saved histogram to {svg_filename}")


def main():
    # algorithms = ["SAC", "AFU"]
    algorithms = ["AFU", "SAC", "IQL", "CALQL"]
    # experiments = ["OnPolicy"]
    experiments = ["OfflineOnlineTransition"]

    env = "Lunar Lander"
    env_dict = {
        "Cartpole Continuous": "CartPoleContinuousStudy-v0",
        "Mountain Car": "MountainCarContinuousStudy-v0",
        "Pendulum": "PendulumStudy-v0",
        "Lunar Lander": "LunarLanderContinuousStudy-v0",
    }
    env_name = env_dict[env]

    colors_dict = {
        "DDPG": "#D00000",
        "SAC": "#38B000",
        "AFU": "#023E8A",
        "IQL": "#F3722C",
        "CALQL": "#2cF372",
    }

    for experiment in experiments:
        experiment_results = {}
        for algo in algorithms:
            filename = f"results/{experiment}-{algo}-{env_name}.pk"
            try:
                with open(filename, "rb") as f:
                    experiment_results[algo] = pickle.load(f)
            except FileNotFoundError:
                print(f"Warning: File {filename} not found")
                continue

        if experiment_results:
            display_results(
                experiment_results,
                colors_dict,
                f"{env} {experiment} Training Progress",
                N=8,
            )

            plot_histograms(experiment_results, colors_dict, experiment, env)


if __name__ == "__main__":
    main()
