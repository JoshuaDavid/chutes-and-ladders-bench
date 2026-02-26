# Chutes & Ladders Bench

**Compare frontier LLMs by Elo — on a child's board game.**

This project is a pedagogical walkthrough of how to build an Elo-based
benchmark from scratch.  The domain (Chutes & Ladders) is intentionally
trivial so the focus stays on the benchmarking infrastructure:

1. Define a **game** with clear rules and tool-use actions.
2. Let two LLMs **play** against each other via function-calling.
3. Record **outcomes** (win / loss / draw / illegal-move).
4. Compute **Elo ratings** from the pairwise results.
5. Generate a **leaderboard chart**.

---

## The Game

| Property | Value |
|---|---|
| Board | 10 × 10, squares 1–100, boustrophedon (snaking) path |
| Spinner | Uniform 1–6 |
| Start | First spin places pawn on square equal to spin value |
| Win | Exact landing on square 100 required |
| Extra turns | None (no bonus for spinning 6) |

### Chutes & Ladders

```
Ladders (↑)          Chutes (↓)
 1 → 38              16 →  6
 4 → 14              47 → 26
 9 → 31              49 → 11
21 → 42              56 → 53
28 → 84              62 → 19
36 → 44              64 → 60
51 → 67              87 → 24
71 → 91              93 → 73
80 → 100             95 → 75
                     98 → 78
```

### Tools available to each player

| Tool | Purpose |
|---|---|
| `spin_spinner` | Spin and get a value 1–6 |
| `move_pawn_to_square` | Move your pawn to `current + spin` |
| `ascend_ladder_to_square` | Take a ladder from your square |
| `descend_chute_to_square` | Slide down a chute from your square |
| `end_turn` | Signal your turn is over |
| `send_message` | Chat with your opponent |
| `forfeit` | Resign (automatic loss) |
| `offer_draw` | Propose a draw |
| `accept_draw` | Accept an offered draw |

**Illegal move = automatic loss.**

---

## Models

| Model | Provider |
|---|---|
| `gpt-4.1-mini` | OpenAI |
| `claude-haiku-4-5-20251001` | Anthropic |
| `google/gemini-3-flash-preview` | OpenRouter |
| `x-ai/grok-4.1-fast` | OpenRouter |
| `qwen/qwen3.5-flash-02-23` | OpenRouter |
| `z-ai/glm-4.7-flash` | OpenRouter |

---

## Quickstart

```bash
# Install
uv sync --all-extras

# Run all pairings (N trials each)
uv run python -m chutes_bench run --trials 5

# Compute Elo from saved results
uv run python -m chutes_bench elo

# Generate leaderboard chart
uv run python -m chutes_bench chart
```

### Environment variables

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
```

---

## Project layout

```
chutes_bench/
  __init__.py
  board.py          # Board state, chutes/ladders map, movement rules
  tools.py          # Tool definitions and action validation
  game.py           # Game runner — orchestrates two LLM players
  elo.py            # Elo calculation from pairwise results
  chart.py          # Matplotlib leaderboard chart
  __main__.py       # CLI entry point
tests/
  test_board.py
  test_tools.py
  test_game.py
  test_elo.py
```

## License

MIT
