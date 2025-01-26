run *ARGS:
    python scripts/train.py

test:
    pytest

format:
    black .

lint:
    flake8
