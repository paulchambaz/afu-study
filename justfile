run *ARGS:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment offpolicy --run 5 --steps 1000000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment hybridpolicy --run 1 --steps 400 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo afu --env pendulum --episodes 2000 --weights ./weights/OnPolicy-AFU-PendulumStudy-v0-weights.pt

dataset:
  python -m scripts.generate_dataset --algo sac --env pendulum --episodes 250 --weights ./weights/OffPolicy-SAC-PendulumStudy-v0-weights.pt --suffix OffPolicy

offline:
  python -m scripts.offline_evaluate --algo afu --env pendulum

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
