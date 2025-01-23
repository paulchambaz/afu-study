# AFU Study

Analysis of the Actor-Free Updates (AFU) reinforcement learning algorithm, focusing on its off-policy properties and handling of offline-online transitions.
This project was developed as part of the M1 AI2D projects course at Sorbonne University (2024-2025).

## Installation

With Nix:

```sh
nix develop
```

With Python virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run experiments:

```sh
just run     # Run main experiments
just test    # Run test suite
```

Generate report:

```sh
just report
```

## Authors

- [Paul Chambaz](https://www.linkedin.com/in/paul-chambaz-17235a158/)
- [Frédéric Li Combeau](https://www.linkedin.com/in/frederic-li-combeau/)
- Supervised by [Olivier Sigaud](https://www.isir.upmc.fr/personnel/sigaud/)

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.
