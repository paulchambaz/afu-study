# Actor Free Critic Update for Off-policy and Offline Learning

This project investigates Actor-Free critic Updates (AFU), a novel reinforcement learning algorithm that decouples critic updates from the actor through value-advantage decomposition, comparing its performance against traditional actor-critic methods in both off-policy learning with randomly sampled data and offline-to-online transition scenarios. Developed as part of the AI2D project course at Sorbonne University, Master AI2D M1, 2024-2025.

## About Actor-Free Critic Updates

Actor-Free critic Updates (AFU) is a recently proposed reinforcement learning algorithm that fundamentally redesigns the actor-critic architecture. Unlike traditional approaches where critic updates depend on actions sampled by the actor, AFU maintains critic updates completely independent from the actor through:

- Value and advantage decomposition with Q(s,a) = V(s) + A(s,a)
- Conditional gradient scaling with an indicator function
- Actor-independent solution to the max-Q problem

The key innovation in AFU lies in how it computes the value and advantage functions. Instead of relying on the actor to estimate the maximum value of Q(s,a) over actions, AFU directly decomposes Q(s,a) into V(s) + A(s,a). This decomposition, combined with a conditional gradient scaling mechanism, allows AFU to solve the maximization problem in Q-learning without depending on the actor.

## Usage

The project uses Nix for dependency management and reproducible builds:

```sh
# Setup with Nix
nix develop
```

Run experiments:

```sh
# Run main experiments
just run

# Run test suite
just test

# Generate report
just report
```

## Project Structure

The project consists of several modules:

- `afu/`: Main implementation of the AFU algorithm and other comparison algorithms
- `scripts/`: Utility scripts for data generation, visualization and benchmarking
- `paper/`: Typst report files with experimental analysis
- `results/`: Experimental results and trained model weights
- `dataset/`: Training datasets for various experimental conditions

## Key Findings

Our experiments across multiple continuous control environments (CartPole, MountainCar, Pendulum, and LunarLander) demonstrate that:

1. In off-policy learning with uniformly sampled random data:

   - All algorithms (including traditional methods like SAC and DDPG) successfully learn from random data in simpler environments with dense rewards
   - No algorithm, including AFU, can learn effectively in the complex LunarLander environment which requires coordinated action sequences
   - The ability to learn from random samples appears more closely tied to reward structure and environment complexity than to algorithm design

2. During offline-to-online transition:

   - AFU demonstrates superior stability during the critical transition from offline to online learning
   - AFU avoids the performance degradation observed in conservative methods like IQL and CAL-QL
   - Quantitative analysis of total regret confirmed AFU's advantage, with significantly smaller deviations from optimal performance

## Authors

- [Paul Chambaz](https://www.linkedin.com/in/paul-chambaz-17235a158/)
- [Frédéric Li Combeau](https://www.linkedin.com/in/frederic-li-combeau/)
- Supervised by [Olivier Sigaud](https://www.isir.upmc.fr/personnel/sigaud/)

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
