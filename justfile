# check style and typing
check-static: lint typecheck

# check style and format
lint:
    uv run -- ruff check --extend-select I .
    uv run -- ruff format --check .

# format code and sort imports
format:
    uv run -- ruff check --select I --fix .
    uv run -- ruff format .

# check static typing annotations
typecheck:
    uv run -- mypy .

# prepare a new release
release version:
    @if [ -n "$(git status --porcelain || echo "dirty")" ]; then echo "repo is dirty!"; exit 1; fi
    sed -i 's/^version = ".*"$/version = "{{ version }}"/g' pyproject.toml
    git add pyproject.toml
    git commit -m "release {{ version }}"
    git tag -m "Release {{ version }}" -a -e "v{{ version }}"
    @echo "check last commit and ammend as necessary, then git push --follow-tags"
