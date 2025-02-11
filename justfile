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

whoworked:
  @git log --all --format='%aN' | sort -u | while read author; do echo -n "$author: "; git log --all --author="$author" --pretty=tformat: --numstat | awk '/.py$|.yaml$|.md$|.nix$|.lock$|.typ$/ {added += $1; deleted += $2} END {print added - deleted, "lines"}'; done
