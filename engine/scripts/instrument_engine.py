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
    d_loop = root / "src/d_loop.c"
    for path in (bridge, p_inter, g_game, d_loop):
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
static int promptfps_snapshot_ready;
static unsigned int promptfps_snapshot_player_damage_dealt;
static unsigned int promptfps_snapshot_player_kills;
static unsigned int promptfps_snapshot_player_pickups;
static unsigned int promptfps_snapshot_level_completions;
static unsigned int promptfps_snapshot_secret_exits;

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
    promptfps_snapshot_ready = 0;
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
    )

    replace_once(
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
        "#include <stdio.h>\n",
        "#include <stdio.h>\n#include <stdlib.h>\n",
        "observation purity allocation include",
    )

    replace_once(
        bridge,
        """EMSCRIPTEN_KEEPALIVE void PromptFPS_SetPaused(int should_pause)
{
    paused = should_pause != 0;
}
""",
        """EMSCRIPTEN_KEEPALIVE void PromptFPS_SetPaused(int should_pause)
{
    paused = should_pause != 0;
}

static int *promptfps_observation_line_validcount;
static int promptfps_observation_line_capacity;

static boolean PromptFPS_CheckSightPure(mobj_t *from, mobj_t *to)
{
    int *next;
    int saved_validcount;
    int result;
    int i;

    if (numlines > promptfps_observation_line_capacity)
    {
        next = realloc(promptfps_observation_line_validcount,
                       sizeof(int) * (size_t) numlines);
        if (next == NULL)
            return P_CheckSight(from, to);
        promptfps_observation_line_validcount = next;
        promptfps_observation_line_capacity = numlines;
    }

    saved_validcount = validcount;
    for (i = 0; i < numlines; ++i)
        promptfps_observation_line_validcount[i] = lines[i].validcount;

    result = P_CheckSight(from, to);

    validcount = saved_validcount;
    for (i = 0; i < numlines; ++i)
        lines[i].validcount = promptfps_observation_line_validcount[i];

    return result;
}

extern void G_PromptFPSSaveSnapshot(void);
extern void G_PromptFPSLoadSnapshot(void);
extern void D_PromptFPSSaveLoopState(void);
extern void D_PromptFPSRestoreLoopState(void);
extern int D_PromptFPSStepTics(int count);

EMSCRIPTEN_KEEPALIVE int PromptFPS_StepTics(int controls, int count)
{
    int ran;
    if (count < 0)
        count = 0;
    if (count > 256)
        count = 256;
    PromptFPS_SetControls(controls);
    ran = D_PromptFPSStepTics(count);
    PromptFPS_SetControls(0);
    return ran;
}

EMSCRIPTEN_KEEPALIVE int PromptFPS_SaveSnapshot(void)
{
    PromptFPS_SetControls(0);
    D_PromptFPSSaveLoopState();
    G_PromptFPSSaveSnapshot();
    promptfps_snapshot_player_damage_dealt = promptfps_player_damage_dealt;
    promptfps_snapshot_player_kills = promptfps_player_kills;
    promptfps_snapshot_player_pickups = promptfps_player_pickups;
    promptfps_snapshot_level_completions = promptfps_level_completions;
    promptfps_snapshot_secret_exits = promptfps_secret_exits;
    promptfps_snapshot_ready = 1;
    return 1;
}

EMSCRIPTEN_KEEPALIVE int PromptFPS_RestoreSnapshot(void)
{
    if (!promptfps_snapshot_ready)
        return 0;
    PromptFPS_SetControls(0);
    // Restore loop counters before loading so G_InitNew observes the original
    // gametic, then restore again after load to undo any incidental changes.
    D_PromptFPSRestoreLoopState();
    G_PromptFPSLoadSnapshot();
    D_PromptFPSRestoreLoopState();
    promptfps_player_damage_dealt = promptfps_snapshot_player_damage_dealt;
    promptfps_player_kills = promptfps_snapshot_player_kills;
    promptfps_player_pickups = promptfps_snapshot_player_pickups;
    promptfps_level_completions = promptfps_snapshot_level_completions;
    promptfps_secret_exits = promptfps_snapshot_secret_exits;
    return 1;
}

EMSCRIPTEN_KEEPALIVE int PromptFPS_HasSnapshot(void)
{
    return promptfps_snapshot_ready;
}
""",
        "snapshot bridge exports",
    )

    replace_once(
        bridge,
        'P_CheckSight(player, mo) ? "true" : "false"',
        'PromptFPS_CheckSightPure(player, mo) ? "true" : "false"',
        "pure entity visibility query",
    )
    replace_once(
        bridge,
        '!P_CheckSight(p->mo, mo)',
        '!PromptFPS_CheckSightPure(p->mo, mo)',
        "pure enemy visibility filter",
    )
    replace_once(
        bridge,
        '!P_CheckSight(p->mo, mo)',
        '!PromptFPS_CheckSightPure(p->mo, mo)',
        "pure pickup visibility filter",
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
        'extern int prndindex;\n'
        'extern int validcount;\n'
        '#endif\n\n'
        '#define SAVEGAMESIZE',
        "level recorder declaration",
    )

    replace_once(
        g_game,
        """//
// G_InitNew
// Can be called by the startup code or the menu task,
""",
        """#if defined(__EMSCRIPTEN__)
#define PROMPTFPS_MAX_SNAPSHOT_MOBJS 8192

static int promptfps_snapshot_rndindex;
static int promptfps_snapshot_prndindex;
static int promptfps_snapshot_paused;
static int promptfps_snapshot_turnheld;
static int promptfps_snapshot_next_weapon;
static int promptfps_snapshot_mobj_count;
static int promptfps_snapshot_target_index[PROMPTFPS_MAX_SNAPSHOT_MOBJS];
static int promptfps_snapshot_tracer_index[PROMPTFPS_MAX_SNAPSHOT_MOBJS];
static int promptfps_snapshot_player_attacker_index[MAXPLAYERS];
static int *promptfps_snapshot_sector_soundtarget_index;
static int *promptfps_snapshot_sector_validcount;
static int *promptfps_snapshot_sector_soundtraversed;
static int *promptfps_snapshot_line_validcount;
static int promptfps_snapshot_sector_capacity;
static int promptfps_snapshot_line_capacity;
static int promptfps_snapshot_validcount;

static int PromptFPS_MobjIndex(mobj_t *needle)
{
    thinker_t *th;
    int index = 0;

    if (needle == NULL)
        return -1;

    for (th = thinkercap.next; th != &thinkercap; th = th->next)
    {
        if (th->function.acp1 != (actionf_p1) P_MobjThinker)
            continue;
        if ((mobj_t *) th == needle)
            return index;
        ++index;
    }

    return -1;
}

static mobj_t *PromptFPS_MobjAtIndex(int wanted)
{
    thinker_t *th;
    int index = 0;

    if (wanted < 0)
        return NULL;

    for (th = thinkercap.next; th != &thinkercap; th = th->next)
    {
        if (th->function.acp1 != (actionf_p1) P_MobjThinker)
            continue;
        if (index == wanted)
            return (mobj_t *) th;
        ++index;
    }

    return NULL;
}

static int PromptFPS_CaptureSnapshotLinks(void)
{
    thinker_t *th;
    int i;
    int index = 0;

    if (numsectors > promptfps_snapshot_sector_capacity)
    {
        int *next;

        next = realloc(promptfps_snapshot_sector_soundtarget_index,
                       sizeof(int) * (size_t) numsectors);
        if (next == NULL)
            return 0;
        promptfps_snapshot_sector_soundtarget_index = next;

        next = realloc(promptfps_snapshot_sector_validcount,
                       sizeof(int) * (size_t) numsectors);
        if (next == NULL)
            return 0;
        promptfps_snapshot_sector_validcount = next;

        next = realloc(promptfps_snapshot_sector_soundtraversed,
                       sizeof(int) * (size_t) numsectors);
        if (next == NULL)
            return 0;
        promptfps_snapshot_sector_soundtraversed = next;
        promptfps_snapshot_sector_capacity = numsectors;
    }

    if (numlines > promptfps_snapshot_line_capacity)
    {
        int *next = realloc(promptfps_snapshot_line_validcount,
                            sizeof(int) * (size_t) numlines);
        if (next == NULL)
            return 0;
        promptfps_snapshot_line_validcount = next;
        promptfps_snapshot_line_capacity = numlines;
    }

    for (i = 0; i < MAXPLAYERS; ++i)
        promptfps_snapshot_player_attacker_index[i] =
            PromptFPS_MobjIndex(players[i].attacker);

    promptfps_snapshot_validcount = validcount;

    for (i = 0; i < numsectors; ++i)
    {
        promptfps_snapshot_sector_soundtarget_index[i] =
            PromptFPS_MobjIndex(sectors[i].soundtarget);
        promptfps_snapshot_sector_validcount[i] = sectors[i].validcount;
        promptfps_snapshot_sector_soundtraversed[i] = sectors[i].soundtraversed;
    }

    for (i = 0; i < numlines; ++i)
        promptfps_snapshot_line_validcount[i] = lines[i].validcount;

    for (th = thinkercap.next; th != &thinkercap; th = th->next)
    {
        mobj_t *mobj;

        if (th->function.acp1 != (actionf_p1) P_MobjThinker)
            continue;
        if (index >= PROMPTFPS_MAX_SNAPSHOT_MOBJS)
            return 0;

        mobj = (mobj_t *) th;
        promptfps_snapshot_target_index[index] = PromptFPS_MobjIndex(mobj->target);
        promptfps_snapshot_tracer_index[index] = PromptFPS_MobjIndex(mobj->tracer);
        ++index;
    }

    promptfps_snapshot_mobj_count = index;
    return 1;
}

static int PromptFPS_RestoreSnapshotLinks(void)
{
    thinker_t *th;
    int i;
    int index = 0;

    for (th = thinkercap.next; th != &thinkercap; th = th->next)
    {
        mobj_t *mobj;

        if (th->function.acp1 != (actionf_p1) P_MobjThinker)
            continue;
        if (index >= promptfps_snapshot_mobj_count)
            return 0;

        mobj = (mobj_t *) th;
        mobj->target = PromptFPS_MobjAtIndex(promptfps_snapshot_target_index[index]);
        mobj->tracer = PromptFPS_MobjAtIndex(promptfps_snapshot_tracer_index[index]);
        ++index;
    }

    if (index != promptfps_snapshot_mobj_count)
        return 0;

    for (i = 0; i < MAXPLAYERS; ++i)
        players[i].attacker =
            PromptFPS_MobjAtIndex(promptfps_snapshot_player_attacker_index[i]);

    if (numsectors > promptfps_snapshot_sector_capacity
        || numlines > promptfps_snapshot_line_capacity)
        return 0;

    validcount = promptfps_snapshot_validcount;

    for (i = 0; i < numsectors; ++i)
    {
        sectors[i].soundtarget =
            PromptFPS_MobjAtIndex(promptfps_snapshot_sector_soundtarget_index[i]);
        sectors[i].validcount = promptfps_snapshot_sector_validcount[i];
        sectors[i].soundtraversed = promptfps_snapshot_sector_soundtraversed[i];
    }

    for (i = 0; i < numlines; ++i)
        lines[i].validcount = promptfps_snapshot_line_validcount[i];

    return 1;
}

void G_PromptFPSSaveSnapshot(void)
{
    if (!PromptFPS_CaptureSnapshotLinks())
        return;
    promptfps_snapshot_rndindex = rndindex;
    promptfps_snapshot_prndindex = prndindex;
    promptfps_snapshot_paused = paused;
    promptfps_snapshot_turnheld = turnheld;
    promptfps_snapshot_next_weapon = next_weapon;
    savegameslot = 7;
    M_StringCopy(savedescription, "PromptFPS snapshot", sizeof(savedescription));
    sendsave = false;
    G_DoSaveGame();
}

void G_PromptFPSLoadSnapshot(void)
{
    G_LoadGame(P_SaveGameFile(7));
    G_DoLoadGame();
    rndindex = promptfps_snapshot_rndindex;
    prndindex = promptfps_snapshot_prndindex;
    paused = promptfps_snapshot_paused;
    turnheld = promptfps_snapshot_turnheld;
    next_weapon = promptfps_snapshot_next_weapon;
    memset(gamekeydown, 0, sizeof(gamekeydown));
    joyxmove = joyymove = joystrafemove = 0;
    mousex = mousey = 0;
    sendpause = sendsave = false;
    PromptFPS_RestoreSnapshotLinks();
}
#endif

//
// G_InitNew
// Can be called by the startup code or the menu task,
""",
        "counterfactual snapshot engine hooks",
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

    insert_before_unique_line(
        d_loop,
        "void D_RegisterLoopCallbacks",
        """#if defined(__EMSCRIPTEN__)
static int promptfps_snapshot_maketic;
static int promptfps_snapshot_recvtic;
static int promptfps_snapshot_gametic;
static int promptfps_snapshot_skiptics;
static int promptfps_snapshot_lasttime;
static boolean promptfps_snapshot_singletics;
static ticcmd_set_t promptfps_snapshot_ticdata[BACKUPTICS];
static boolean promptfps_snapshot_local_playeringame[NET_MAXPLAYERS];

void D_PromptFPSSaveLoopState(void)
{
    promptfps_snapshot_maketic = maketic;
    promptfps_snapshot_recvtic = recvtic;
    promptfps_snapshot_gametic = gametic;
    promptfps_snapshot_skiptics = skiptics;
    promptfps_snapshot_lasttime = lasttime;
    promptfps_snapshot_singletics = singletics;
    memcpy(promptfps_snapshot_ticdata, ticdata, sizeof(ticdata));
    memcpy(promptfps_snapshot_local_playeringame, local_playeringame,
           sizeof(local_playeringame));
}

void D_PromptFPSRestoreLoopState(void)
{
    maketic = promptfps_snapshot_maketic;
    recvtic = promptfps_snapshot_recvtic;
    gametic = promptfps_snapshot_gametic;
    skiptics = promptfps_snapshot_skiptics;
    lasttime = promptfps_snapshot_lasttime;
    singletics = promptfps_snapshot_singletics;
    memcpy(ticdata, promptfps_snapshot_ticdata, sizeof(ticdata));
    memcpy(local_playeringame, promptfps_snapshot_local_playeringame,
           sizeof(local_playeringame));
}

int D_PromptFPSStepTics(int count)
{
    boolean old_singletics = singletics;
    int i;

    if (count < 0)
        count = 0;
    if (count > 256)
        count = 256;

    singletics = true;
    for (i = 0; i < count; ++i)
        TryRunTics();
    singletics = old_singletics;

    return count;
}
#endif

""",
        "exact tic stepping hook",
    )

    print("doom-classifier telemetry instrumentation applied")


if __name__ == "__main__":
    main()
