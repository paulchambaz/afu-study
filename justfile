run *ARGS:
    python -m scripts.train

test:
    pytest

format:
    black .

lint:
  pylint afu
