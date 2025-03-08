run *ARGS:
  python -m scripts.evaluate --algo afuperrin --env pendulum --experiment onpolicy --run 15 --steps 60000 --trials 100

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
  python -m scripts.evaluate --algo afuperrin --env lunarlander --experiment onpolicy --run 1 --steps 10 --trials 2

fmt:
  ruff format

lint:
  ruff check afu

report:
  typst compile paper/paper.typ
  typst compile paper/pres.typ

copy:
  fd '(oj|cr).*typ' . | xargs cat | wl-copy
