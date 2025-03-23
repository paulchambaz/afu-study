run *ARGS:
  python -m scripts.evaluate --algo afu --env lunarlander --experiment onpolicy --run 5 --steps 1000000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo sac --env swimmer --experiment onpolicy --run 1 --steps 1024 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo sac --env lunarlander --episodes 20 --weights ./weights/OnPolicy-SAC-LunarLanderContinuousStudy-v0-weights.pt

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
