"""Export game data to JSON for the web viewer, with optional B2 upload."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path


def export_games_list(db_path: Path | str) -> list[dict]:
    """Read all games from the DB and return as a list of dicts."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, player_a, player_b, winner, reason, turns, created_at "
        "FROM games ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def export_game_events(db_path: Path | str, game_id: int) -> dict | None:
    """Export all events for a single game as a structured dict.

    Returns ``None`` if the game does not exist.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Game metadata
    game_row = conn.execute(
        "SELECT id, player_a, player_b, winner, reason, turns, created_at "
        "FROM games WHERE id = ?",
        (game_id,),
    ).fetchone()
    if game_row is None:
        conn.close()
        return None

    game = dict(game_row)

    # Turn summaries
    turn_rows = conn.execute(
        "SELECT turn_number, player_idx, start_position, end_position, "
        "spin_value, outcome, actions_count "
        "FROM turns WHERE game_id = ? ORDER BY turn_number",
        (game_id,),
    ).fetchall()
    turns = [dict(r) for r in turn_rows]

    # LLM invocations
    inv_rows = conn.execute(
        "SELECT id, turn_number, player_idx, sequence_in_turn, model_api_id, "
        "response_raw, input_tokens, output_tokens, latency_ms "
        "FROM llm_invocations WHERE game_id = ? "
        "ORDER BY turn_number, sequence_in_turn",
        (game_id,),
    ).fetchall()

    # Tool calls
    tc_rows = conn.execute(
        "SELECT invocation_id, turn_number, player_idx, tool_name, tool_args, "
        "result_ok, result_message, board_before, board_after, "
        "is_winning_move, is_illegal, is_forfeit, is_turn_over "
        "FROM tool_calls WHERE game_id = ? ORDER BY id",
        (game_id,),
    ).fetchall()
    conn.close()

    # Group tool calls by invocation id
    tc_by_inv: dict[int, list[dict]] = {}
    for tc in tc_rows:
        tc_dict = dict(tc)
        inv_id = tc_dict.pop("invocation_id")
        tc_dict["tool_args"] = json.loads(tc_dict["tool_args"])
        tc_dict["board_before"] = json.loads(tc_dict["board_before"])
        tc_dict["board_after"] = json.loads(tc_dict["board_after"])
        for key in ("result_ok", "is_winning_move", "is_illegal", "is_forfeit", "is_turn_over"):
            tc_dict[key] = bool(tc_dict[key])
        tc_dict["type"] = "tool_call"
        tc_by_inv.setdefault(inv_id, []).append(tc_dict)

    # Build flat chronological event list
    events: list[dict] = []
    for inv in inv_rows:
        inv_dict = dict(inv)
        inv_id = inv_dict["id"]

        # Extract assistant text from response
        raw = inv_dict["response_raw"]
        response_raw = json.loads(raw) if raw else {}
        text = _extract_response_text(response_raw)

        if text:
            events.append({
                "type": "llm_response",
                "turn_number": inv_dict["turn_number"],
                "player_idx": inv_dict["player_idx"],
                "model": inv_dict["model_api_id"],
                "text": text,
                "input_tokens": inv_dict["input_tokens"],
                "output_tokens": inv_dict["output_tokens"],
                "latency_ms": inv_dict["latency_ms"],
            })

        for tc in tc_by_inv.get(inv_id, []):
            events.append(tc)

    return {"game": game, "turns": turns, "events": events}


def _extract_response_text(response_raw: dict) -> str | None:
    """Extract text content from an LLM response dict.

    Handles both OpenAI and Anthropic response formats.
    """
    if not response_raw:
        return None

    # OpenAI format: response.choices[0].message.content
    choices = response_raw.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        content = msg.get("content")
        if content:
            return content

    # Anthropic format: response.content[].text
    content_blocks = response_raw.get("content", [])
    if isinstance(content_blocks, list):
        texts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        if texts:
            return "\n".join(texts)

    return None


def generate_all(db_path: Path | str, output_dir: Path) -> list[Path]:
    """Generate games.json and per-game event JSON files.

    Returns a list of all generated file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    events_dir = output_dir / "events"
    events_dir.mkdir(exist_ok=True)

    generated: list[Path] = []

    games = export_games_list(db_path)
    games_path = output_dir / "games.json"
    games_path.write_text(json.dumps(games, indent=2))
    generated.append(games_path)

    for game in games:
        game_id = game["id"]
        data = export_game_events(db_path, game_id)
        if data is None:
            continue
        event_path = events_dir / f"{game_id}.json"
        event_path.write_text(json.dumps(data, indent=2))
        generated.append(event_path)

    return generated


def upload_to_b2(
    files: dict[str, bytes],
    bucket_name: str,
    endpoint_url: str,
    key_id: str,
    app_key: str,
) -> None:
    """Upload files to Backblaze B2 using the S3-compatible API.

    ``files`` maps object keys (e.g. ``"data/games.json"``) to content bytes.
    """
    import boto3  # type: ignore[import-untyped]

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=key_id,
        aws_secret_access_key=app_key,
    )

    for key, content in files.items():
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=content,
            ContentType="application/json",
        )
