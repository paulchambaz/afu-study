import pickle

dataset_files = [
    "dataset/OffPolicyDataset-AFU-PendulumStudy-v0-dataset.pk",
    "dataset/OnPolicyDataset-SAC-PendulumStudy-v0-dataset.pk",
]

for file in dataset_files:
    with open(file, "rb") as f:
        results = pickle.load(f)

    episode_counter = 0
    new_results = []
    for i in range(len(results)):
        if episode_counter % 20 == 0:
            new_results.append(results[i])

        if results[i][4]:
            episode_counter += 1

    print(len(new_results))

    with open(file, "wb") as f:
        pickle.dump(new_results, f)
