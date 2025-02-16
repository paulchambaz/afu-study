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
    plt.show()

def main():
    algorithms = ["DDPG", "SAC", "AFUPerrin"]
    experiments = ["OffPolicy", "OnPolicy"]
    env_name = "CartPoleContinuousStudy-v0"
    
    colors_dict = {
        "DDPG": "#D00000",
        "SAC": "#38B000",
        "AFUPerrin": "#023E8A",
    }
    
    # Create a figure for each experiment type
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
                
        if experiment_results:  # Only create plot if we have data
            display_results(
                experiment_results, 
                colors_dict, 
                f"{experiment} Training Progress",
                N=4 if experiment == "OnPolicy" else 1
            )

if __name__ == "__main__":
    main()
