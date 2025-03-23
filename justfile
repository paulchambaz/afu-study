run *ARGS:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment offpolicy --run 5 --steps 200000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment offpolicy --run 1 --steps 1024 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo afu --env lunarlander --episodes 2000 --weights ./weights/OnPolicy-AFU-LunarLanderContinuousStudy-v0-weights.pt

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
