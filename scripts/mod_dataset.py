import pickle

dataset_files = ["dataset/OnPolicyDataset-AFU-LunarLanderContinuousStudy-v0-dataset.pk"]

for file in dataset_files:
    # Load the original dataset
    with open(file, "rb") as f:
        results = pickle.load(f)

    print(f"Original dataset size: {len(results)} transitions")

    # Step 1: Group transitions into episodes
    episodes = []
    current_episode = []

    for transition in results:
        current_episode.append(transition)
        _, _, _, _, done = transition

        if done:
            episodes.append(current_episode)
            current_episode = []

    total_episodes = len(episodes)
    print(f"Found {total_episodes} episodes")

    # Step 2: Calculate how many episodes we need to keep
    target_size = 200000

    # First, calculate the total transitions we'd have with each stride
    stride_results = {}
    for stride in range(1, total_episodes):
        selected = episodes[::stride]
        size = sum(len(ep) for ep in selected)
        stride_results[stride] = size

        # Early exit if we find a good result
        if 200000 <= size <= 210000:
            break

    # Find the stride that gives us just over 200k transitions
    valid_strides = [
        (s, size) for s, size in stride_results.items() if size >= target_size
    ]
    if valid_strides:
        best_stride, best_size = min(valid_strides, key=lambda x: x[1])
    else:
        best_stride = 1
        best_size = sum(len(ep) for ep in episodes)

    # Select episodes with the best stride
    selected_episodes = episodes[::best_stride]

    # Flatten back to transitions
    new_results = []
    for episode in selected_episodes:
        new_results.extend(episode)

    print(f"Selected {len(selected_episodes)} episodes with stride {best_stride}")
    print(f"New dataset size: {len(new_results)} transitions")

    # Save the new dataset
    with open(file + "new", "wb") as f:
        pickle.dump(new_results, f)
