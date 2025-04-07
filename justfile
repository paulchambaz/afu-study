run *ARGS:
  # python -m scripts.evaluate --algo afu --env lunarlander --experiment offpolicynetwork --run 5 --steps 200000 --trials 0
  python -m scripts.evaluate --algo afu --env pendulum --experiment hybridpolicy --run 1 --steps 300000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment hybridpolicy --run 1 --steps 400 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo afu --env pendulum --episodes 2000 --weights ./weights/OnPolicy-AFU-PendulumStudy-v0-weights.pt

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
