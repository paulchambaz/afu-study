run *ARGS:
  python -m scripts.evaluate --algo sac --env pendulum --experiment onpolicy --run 5 --steps 50000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo sac --env pendulum --experiment onpolicy --run 1 --steps 1024 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

benchmark *ARGS:
  python -m scripts.benchmark {{ARGS}}

demo:
  python -m scripts.demo --algo sac --env pendulum --episodes 1 --weights ./weights/OnPolicy-SAC-PendulumStudy-v0-weights.pt

test:
  pytest

quicktest:
  python -m scripts.evaluate --algo afuperrin --env pendulum --experiment onpolicy --run 1 --steps 10 --trials 1

fmt:
  ruff format

lint:
  ruff check afu

report:
  typst compile paper/paper.typ
  typst compile paper/pres.typ

copy:
  fd '(oj|cr).*typ' . | xargs cat | wl-copy
