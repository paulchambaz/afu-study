run *ARGS:
    python -m scripts.train {{ARGS}}

evaluate *ARGS:
    python -m scripts.evaluate {{ARGS}}

plot *ARGS:
    python -m scripts.plot_results {{ARGS}}

benchmark:
  hivemind

demo:
  python -m scripts.demo

test:
    pytest

fmt:
    black .

lint:
  pylint afu

report:
  typst compile paper/paper.typ
  typst compile paper/pres.typ

copy:
  fd '(oj|cr).*typ' . | xargs cat | wl-copy
