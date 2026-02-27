# CLAUDE.md

## Project overview

Pedagogical Elo benchmark: LLMs play Chutes & Ladders via tool-use, results
feed into Elo ratings and a leaderboard chart.

## Development workflow

**Red / Green / Refactor** — always write a failing test before implementing.

```
1. RED    — write a test that fails (or doesn't compile)
2. GREEN  — write the minimum code to make it pass
3. REFACTOR — clean up, then re-run tests
```

## Commands

```bash
# Install / sync deps
uv sync --all-extras

# Run tests
uv run pytest -xvs

# Run a single test file
uv run pytest -xvs tests/test_board.py

# Type-check
uv run pyright chutes_bench/

# Run the CLI
uv run python -m chutes_bench --help
```

## Project structure

- `chutes_bench/` — all source code lives here
- `tests/` — pytest tests, mirror the source layout
- `results/` — game logs (gitignored, created at runtime)

## Conventions

- Python 3.11+
- Use `dataclass` or plain dicts — no heavy frameworks
- Type hints on all public functions
- Tests use plain `pytest` (no unittest classes)
- Keep modules small and focused — one concept per file

## Key domain rules

- Board: squares 1–100, boustrophedon layout
- Spinner: uniform 1–6
- First spin places you on that square number
- Must land exactly on 100 to win
- No extra turn for spinning 6
- Landing on a chute/ladder base triggers it automatically
- Illegal move = automatic loss
- Forfeit = automatic loss
- Draw only by mutual agreement (offer + accept)
