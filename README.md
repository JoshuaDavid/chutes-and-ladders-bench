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

# Run all pairings (N trials each) — resumes automatically if interrupted
uv run python -m chutes_bench run --trials 5

# Compute Elo from saved results
uv run python -m chutes_bench elo

# Generate leaderboard chart
uv run python -m chutes_bench chart
```

Results are stored in `results/benchmark.db` (SQLite, gitignored). You can
stop and restart `run` at any time — it picks up where it left off.

### Results database

The database is stored in Backblaze B2 (not in git — it's ~4 MB/game).

```bash
# Download the latest results
python scripts/download_db.py

# Upload after running new games (requires B2 credentials)
python scripts/upload_db.py
```

The DB contains detailed transcripts for every game: full LLM
request/response payloads, tool calls with validation results, board state
snapshots, and token usage. See [Database schema](#database-schema) below.

### Environment variables

```
# LLM providers (required for running games)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...

# Backblaze B2 (required only for uploading results)
BACKBLAZE_KEY_ID=...
BACKBLAZE_APPLICATION_KEY=...
BACKBLAZE_CHUTES_BUCKET_NAME=...
```

---

## Project layout

```
chutes_bench/
  __init__.py
  board.py          # Board state, chutes/ladders map, movement rules
  tools.py          # Tool definitions and action validation
  game.py           # Game runner — orchestrates two LLM players
  players.py        # LLM player adapters (OpenAI, Anthropic, OpenRouter)
  invocation.py     # LLMInvocation dataclass for capturing API calls
  elo.py            # Elo calculation from pairwise results
  chart.py          # Matplotlib leaderboard chart
  persistence.py    # SQLite storage for pause/resume/parallelization
  __main__.py       # CLI entry point
scripts/
  download_db.py    # Download benchmark.db from Backblaze B2
  upload_db.py      # Upload benchmark.db to Backblaze B2
tests/
  test_board.py
  test_tools.py
  test_moves.py
  test_game.py
  test_elo.py
  test_persistence.py
  test_invocation_capture.py
  test_detailed_persistence.py
  test_game_logging.py
  test_persist_game_log.py
```

---

## Database schema

Every game records full transcripts across these tables:

| Table | Purpose |
|---|---|
| `models` | Model registry (api_id, display_name, provider) |
| `games` | Game outcomes (players, winner, reason, turn count) |
| `pairings` | Scheduled matchups for pause/resume |
| `turns` | Per-turn summary (positions, spin value, outcome) |
| `llm_invocations` | Every LLM API call (full request messages, raw response, tokens, latency) |
| `tool_calls` | Every tool call (name, args, validation result, board state before/after) |

Example queries:

```sql
-- Full transcript for a game
SELECT turn_number, tool_name, tool_args, result_message, board_before, board_after
FROM tool_calls WHERE game_id = 4 ORDER BY id;

-- Token cost per model
SELECT model_api_id, SUM(input_tokens) as total_in, SUM(output_tokens) as total_out
FROM llm_invocations GROUP BY model_api_id;

-- Illegal move rate
SELECT model_api_id,
       COUNT(*) FILTER (WHERE is_illegal) as illegal,
       COUNT(*) as total
FROM tool_calls tc
JOIN llm_invocations li ON tc.invocation_id = li.id
GROUP BY model_api_id;
```

## License

MIT
