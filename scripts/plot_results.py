import pickle
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(data):
    data = np.array(data)
    q1, q3 = np.percentile(data, [25, 75])
    mask = (data >= q1) & (data <= q3)
    iqm = np.mean(data[mask])
    return q1, q3, iqm


def display_auc(results_dict, colors_dict, N=4):
    plt.figure(figsize=(6, 6))

    # Data for the plot
    algo_names = []
    q1_values = []
    iqm_values = []
    q3_values = []
    colors_list = []

    # Collect the data
    for algo, results in results_dict.items():
        algo_names.append(algo)
        colors_list.append(colors_dict[algo])

        max_q3 = float("-inf")
        rewards_data = results["rewards"]

        # Find max values
        for timestep in sorted(rewards_data.keys()):
            data = np.array(rewards_data[timestep])
            _, q3, _ = compute_stats(data)
            if q3 > max_q3:
                max_q3 = q3

        # max_q3 = 300

        # Calculate total regret
        total_regret_iqm = 0
        total_regret_q1 = 0
        total_regret_q3 = 0
        for timestep in sorted(rewards_data.keys())[2000:]:
            data = np.array(rewards_data[timestep])
            q1, q3, iqm = compute_stats(data)
            regret_iqm = max_q3 - iqm
            total_regret_iqm += regret_iqm
            regret_q1 = max_q3 - q1
            total_regret_q1 += regret_q1
            regret_q3 = max_q3 - q3
            total_regret_q3 += regret_q3

        # Store the values for the plot
        q1_values.append(total_regret_q1)
        iqm_values.append(total_regret_iqm)
        q3_values.append(total_regret_q3)

        # Print the values
        print(algo, total_regret_q1, total_regret_iqm, total_regret_q3)

    # Create positions for the plots
    positions = np.arange(len(algo_names))
    width = 0.2  # Width of the bars

    # Create a custom visualization that resembles a box plot
    # but only shows q1, iqm and q3
    for i, algo in enumerate(algo_names):
        # Plot vertical line from q1 to q3
        width_rect = 0.15  # Width of rectangle
        plt.gca().add_patch(
            plt.Rectangle(
                (i - width_rect / 2, q1_values[i]),  # (x, y) of bottom left corner
                width_rect,  # width
                q3_values[i] - q1_values[i],  # height
                color=colors_list[i],
                alpha=0.4,
            )
        )

        # Plot horizontal line at q1
        plt.plot(
            [i - width / 2, i + width / 2],
            [q1_values[i], q1_values[i]],
            color=colors_list[i],
            linewidth=2,
        )

        # Plot horizontal line at q3
        plt.plot(
            [i - width / 2, i + width / 2],
            [q3_values[i], q3_values[i]],
            color=colors_list[i],
            linewidth=2,
        )

        # Plot marker for iqm (median)
        plt.plot(
            i,
            iqm_values[i],
            "o",
            color="black",
            markersize=8,
            markeredgecolor=colors_list[i],
            markerfacecolor=colors_list[i],
        )

    # Set the x-axis labels and ticks
    plt.xticks(positions, algo_names)
    plt.xlabel("Algorithms")
    plt.ylabel("Total Regret")
    plt.grid(True, axis="y", alpha=0.3)

    # Add a legend explaining the elements
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2, label="Q1 to Q3 Range"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markersize=8,
            markerfacecolor="gray",
            label="IQM Value",
        ),
    ]
    plt.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig("auc_comparison.svg", format="svg")
    # plt.show()


def display_results(results_dict, colors_dict, title, N=4):
    interval = 100
    plt.figure(figsize=(6, 6))

    for algo, results in results_dict.items():
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
            label=f"{algo}",
        )
        plt.fill_between(
            timesteps[plot_indices],
            q1_values[plot_indices],
            q3_values[plot_indices],
            alpha=0.2,
            color=colors_dict[algo],
        )

    plt.axvline(
        x=200_000,
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel("Training Steps")
    plt.ylabel("Return")
    # plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    from matplotlib.ticker import FuncFormatter

    def format_func(x, pos):
        return f"{int(x / 1000)}k"

    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))

    plt.tight_layout(pad=0.5)

    plt.savefig(f"{title}.svg", format="svg", bbox_inches="tight")


def plot_histograms(experiment_results, colors_dict, experiment, env):
    plt.figure(figsize=(6, 6))

    algo_rewards = {}
    min_reward = float("inf")
    max_reward = float("-inf")

    for algo, results in experiment_results.items():
        rewards_data = results["rewards"]
        all_timesteps = sorted(list(rewards_data.keys()))
        last_timesteps = all_timesteps[-int(len(all_timesteps) * 0.01) :]
        all_rewards = []
        for timestep in last_timesteps:
            all_rewards.extend(rewards_data[timestep])
        algo_rewards[algo] = all_rewards

        min_reward = min(min_reward, min(all_rewards))
        max_reward = max(max_reward, max(all_rewards))

    bins = 15

    bin_edges = np.linspace(min_reward, max_reward, bins + 1)

    plt.hist(
        [algo_rewards[algo] for algo in algo_rewards],
        bins=bin_edges,
        label=list(algo_rewards.keys()),
        color=[colors_dict[algo] for algo in algo_rewards],
        alpha=0.7,
        histtype="bar",
        rwidth=0.8,
    )

    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig(f"{env}_{experiment}_histograms.svg", format="svg")


def main():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "axes.labelsize": 20,
            "font.size": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
        }
    )

    # algorithms = ["SAC", "AFU", "DDPG"]
    algorithms = ["AFU", "SAC", "IQL", "CALQL"]
    # experiments = ["OffPolicy", "OnPolicy"]
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
        "DDPG": "#6ca247",
        "SAC": "#d66b6a",
        "AFU": "#5591e1",
        "IQL": "#39a985",
        "CALQL": "#ad75ca",
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
            # display_results(
            #     experiment_results,
            #     colors_dict,
            #     f"{env} {experiment} Training Progress",
            #     N=8,
            # )

            display_auc(
                experiment_results,
                colors_dict,
                N=8,
            )

            # plot_histograms(experiment_results, colors_dict, experiment, env)


if __name__ == "__main__":
    main()
