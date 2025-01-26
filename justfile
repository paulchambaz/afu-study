run *ARGS:
    python scripts.train

test:
    pytest

format:
    black .

lint:
    flake8
