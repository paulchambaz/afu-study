run *ARGS:
  # python -m scripts.evaluate --algo calql --env lunarlander --experiment offlineonline --run 5 --steps 400000 --trials 0
  # python -m scripts.evaluate --algo iql --env lunarlander --experiment offlineonline --run 5 --steps 400000 --trials 0
  # python -m scripts.evaluate --algo sac --env lunarlander --experiment offlineonline --run 5 --steps 400000 --trials 0
  python -m scripts.evaluate --algo afu --env lunarlander --experiment offlineonline --run 5 --steps 400000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo afu --env pendulum --experiment onpolicy --run 1 --steps 500 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo afu --env pendulum --episodes 2000 --weights ./weights/OffPolicy-AFU-PendulumStudy-v0-weights.pt

dataset:
  python -m scripts.mod_dataset

figure:
  python -m scripts.figure --algo calql --env pendulum --weights ./weights/OfflineOnlineTransition-CALQL-PendulumStudy-v0-weights.pt

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
