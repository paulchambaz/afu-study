run *ARGS:
  python -m scripts.evaluate --algo afu --env pendulum --experiment onpolicy --run 11 --steps 200000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo afu --env pendulum --experiment offpolicy --run 1 --steps 1024 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

benchmark *ARGS:
  python -m scripts.benchmark {{ARGS}}

demo:
  python -m scripts.demo

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
