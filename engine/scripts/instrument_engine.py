#!/usr/bin/env python3
"""Apply doom-classifier's telemetry ABI to the pinned Chocolate Doom source."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one {label} anchor, found {count}")
    path.write_text(text.replace(old, new, 1))


def insert_before_unique_line(path: Path, token: str, insertion: str, label: str) -> None:
    lines = path.read_text().splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if token in line]
    if len(matches) != 1:
        raise SystemExit(f"{path}: expected exactly one {label} line containing {token!r}, found {len(matches)}")
    lines.insert(matches[0], insertion)
    path.write_text("".join(lines))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: instrument_engine.py <chocolate-doom-root>")

    root = Path(sys.argv[1]).resolve()
    bridge = root / "src/doom/browser_doom_bridge.c"
    p_inter = root / "src/doom/p_inter.c"
    g_game = root / "src/doom/g_game.c"
    for path in (bridge, p_inter, g_game):
        if not path.is_file():
            raise SystemExit(f"missing expected source file: {path}")

    replace_once(
        bridge,
        "static int promptfps_controls;\n",
        """static int promptfps_controls;

static unsigned int promptfps_player_damage_dealt;
static unsigned int promptfps_player_kills;
static unsigned int promptfps_player_pickups;
static unsigned int promptfps_level_completions;
static unsigned int promptfps_secret_exits;

void PromptFPS_RecordPlayerDamage(int damage)
{
    if (damage > 0)
        promptfps_player_damage_dealt += (unsigned int) damage;
}

void PromptFPS_RecordPlayerKill(void)
{
    ++promptfps_player_kills;
}

void PromptFPS_RecordPlayerPickup(void)
{
    ++promptfps_player_pickups;
}

void PromptFPS_RecordLevelComplete(int secret_exit)
{
    ++promptfps_level_completions;
    if (secret_exit)
        ++promptfps_secret_exits;
}

static void PromptFPS_ResetEvents(void)
{
    promptfps_player_damage_dealt = 0;
    promptfps_player_kills = 0;
    promptfps_player_pickups = 0;
    promptfps_level_completions = 0;
    promptfps_secret_exits = 0;
}
""",
        "bridge event state",
    )

    insert_before_unique_line(
        bridge,
        "visible_enemies",
        '        "\\"events\\":{\\"player_damage_dealt\\":%u,\\"player_kills\\":%u,"\n'
        '        "\\"player_pickups\\":%u,\\"level_completions\\":%u,\\"secret_exits\\":%u},"\n',
        "observation event JSON",
    )    replace_once(
        bridge,
        'gametic, gamestate, paused ? "true" : "false", promptfps_controls);',
        'gametic, gamestate, paused ? "true" : "false", promptfps_controls,\n'
        '        promptfps_player_damage_dealt, promptfps_player_kills,\n'
        '        promptfps_player_pickups, promptfps_level_completions,\n'
        '        promptfps_secret_exits);',
        "observation event arguments",
    )
    replace_once(
        bridge,
        "    unsigned int i;\n    G_InitNew(sk_baby, 1, 1);\n",
        "    unsigned int i;\n    PromptFPS_ResetEvents();\n    G_InitNew(sk_baby, 1, 1);\n",
        "set-start event reset",
    )

    replace_once(
        p_inter,
        '#include "p_inter.h"\n\n\n#define BONUSADD',
        '#include "p_inter.h"\n\n'
        '#if defined(__EMSCRIPTEN__)\n'
        'extern void PromptFPS_RecordPlayerDamage(int damage);\n'
        'extern void PromptFPS_RecordPlayerKill(void);\n'
        'extern void PromptFPS_RecordPlayerPickup(void);\n'
        '#endif\n\n'
        '#define BONUSADD',
        "interaction recorder declarations",
    )

    replace_once(
        p_inter,
        "    if (special->flags & MF_COUNTITEM)\n\tplayer->itemcount++;\n",
        "#if defined(__EMSCRIPTEN__)\n"
        "    if (player == &players[consoleplayer])\n"
        "        PromptFPS_RecordPlayerPickup();\n"
        "#endif\n\n"
        "    if (special->flags & MF_COUNTITEM)\n\tplayer->itemcount++;\n",
        "successful pickup recorder",
    )

    replace_once(
        p_inter,
        "    if (source && source->player)\n    {\n\t// count for intermission\n",
        "#if defined(__EMSCRIPTEN__)\n"
        "    if (source && source->player == &players[consoleplayer]\n"
        "        && (target->flags & MF_COUNTKILL))\n"
        "        PromptFPS_RecordPlayerKill();\n"
        "#endif\n\n"
        "    if (source && source->player)\n    {\n\t// count for intermission\n",
        "player-attributed kill recorder",
    )

    replace_once(
        p_inter,
        "    // do the damage\t\n    target->health -= damage;\t\n",
        "#if defined(__EMSCRIPTEN__)\n"
        "    if (source && source->player == &players[consoleplayer]\n"
        "        && target != source && target->player == NULL\n"
        "        && (target->flags & MF_COUNTKILL))\n"
        "    {\n"
        "        int attributed_damage = damage;\n"
        "        if (attributed_damage > target->health)\n"
        "            attributed_damage = target->health;\n"
        "        if (attributed_damage > 0)\n"
        "            PromptFPS_RecordPlayerDamage(attributed_damage);\n"
        "    }\n"
        "#endif\n\n"
        "    // do the damage\t\n    target->health -= damage;\t\n",
        "player-attributed damage recorder",
    )

    replace_once(
        g_game,
        '#include "g_game.h"\n\n\n#define SAVEGAMESIZE',
        '#include "g_game.h"\n\n'
        '#if defined(__EMSCRIPTEN__)\n'
        'extern void PromptFPS_RecordLevelComplete(int secret_exit);\n'
        '#endif\n\n'
        '#define SAVEGAMESIZE',
        "level recorder declaration",
    )

    replace_once(
        g_game,
        "void G_DoCompleted (void) \n{ \n    int             i; \n",
        "void G_DoCompleted (void) \n{ \n    int             i; \n\n"
        "#if defined(__EMSCRIPTEN__)\n"
        "    PromptFPS_RecordLevelComplete(secretexit ? 1 : 0);\n"
        "#endif\n",
        "level completion recorder",
    )

    print("doom-classifier telemetry instrumentation applied")


if __name__ == "__main__":
    main()
