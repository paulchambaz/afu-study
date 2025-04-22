run *ARGS:
  python -m scripts.evaluate --algo sac --env pendulum --experiment onpolicy --run 5 --steps 100000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment hybridpolicy --run 1 --steps 400 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo sac --env pendulum --episodes 2000 --weights ./weights/OnPolicy-SAC-PendulumStudy-v0-weights.pt

generate:
  python -m scripts.generate_dataset --algo sac --env pendulum --episodes 100 --weights ./weights/OnPolicy-SAC-PendulumStudy-v0-weights.pt --output ./dataset/OnPolicy-SAC-PendulumStudy-v0-data.pk

dqn:
  python -m scripts.dqn_test --episodes 500

test:
  pytest

fmt:
  ruff format

lint:
  ruff check afu

report:
  typst compile paper/paper.typ
  typst compile paper/pres.typ

copy:
  fd '(oj|cr).*typ' . | xargs cat | wl-copy
