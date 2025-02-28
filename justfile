run *ARGS:
  python -m scripts.evaluate --algo afuperrin --env pendulum --experiment onpolicy --run 15 --steps 200000 --trials 50

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

fmt:
  ruff format

lint:
  ruff check afu

report:
  typst compile paper/paper.typ
  typst compile paper/pres.typ

copy:
  fd '(oj|cr).*typ' . | xargs cat | wl-copy
