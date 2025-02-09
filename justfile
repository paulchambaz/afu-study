run *ARGS:
    python -m scripts.train

demo:
  python -m scripts.demo

test:
    pytest

fmt:
    black .

lint:
  pylint afu
