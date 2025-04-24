run *ARGS:
  python -m scripts.evaluate --algo calql --env pendulum --experiment offline --run 5 --steps 200000 --trials 0

run-tiny:
  python -m scripts.evaluate --algo iql --env pendulum --experiment offline --run 1 --steps 2000 --trials 0

evaluate *ARGS:
  python -m scripts.evaluate {{ARGS}}

plot *ARGS:
  python -m scripts.plot_results {{ARGS}}

demo:
  python -m scripts.demo --algo afu --env pendulum --episodes 2000 --weights ./weights/OffPolicy-AFU-PendulumStudy-v0-weights.pt

generate:
  python -m scripts.generate_dataset --algo afu --env pendulum --episodes 250 --weights ./weights/OnPolicy-AFU-PendulumStudy-v0-weights.pt --output ./dataset/OnPolicy-AFU-PendulumStudy-v0-data.pk

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
