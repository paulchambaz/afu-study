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

whoworked:
  @git log --all --format='%aN' | sort -u | while read author; do echo -n "$author: "; git log --all --author="$author" --pretty=tformat: --numstat | awk '/.py$|.yaml$|.md$|.nix$|.lock$|.typ$/ {added += $1; deleted += $2} END {print added - deleted, "lines"}'; done

copy:
  fd '(oj|cr).*typ' . | xargs cat | wl-copy
