"""Gymnasium wrapper for Prince of Persia (1989), drives SDLPoP through ctypes"""
import ctypes
from ctypes import c_int, c_short, c_ushort, c_uint8, c_int8, c_byte, c_uint
import os
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

FRAME_SKIP = 4
N_ACTIONS = 12
ROWS, COLS = 3, 10
N_CH = 11
FRAME_STACK = 4
PHYS_DIM = 18
ACT_HIST = 5

# tile ids we actually touch below, rest of the types.h enum isn't needed here
T_SPIKES = 2
T_GATE = 4
T_STUCK = 5
T_CLOSER = 6
T_POTION = 10
T_LOOSE = 11
T_OPENER = 15
T_EXIT_LEFT = 16
T_EXIT_RIGHT = 17
T_CHOMPER = 18
T_SWORD = 22

_CX = np.linspace(-1, 1, COLS, dtype=np.float32)[None, :].repeat(ROWS, 0).copy()
_CY = np.linspace(-1, 1, ROWS, dtype=np.float32)[:, None].repeat(COLS, 1).copy()

_BTN = np.full(32, -1.0, np.float32)
_BTN[T_STUCK] = 1 / 3 * 2 - 1
_BTN[T_CLOSER] = 2 / 3 * 2 - 1
_BTN[T_OPENER] = 1.0

_POT = np.zeros(8, np.float32)
_POT[0] = 1 / 7 * 2 - 1
_POT[1] = 2 / 7 * 2 - 1
_POT[2] = 4 / 7 * 2 - 1
_POT[3] = 5 / 7 * 2 - 1
_POT[4] = 3 / 7 * 2 - 1

_lib = None
_engine_init = False


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    so = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SDLPoP', 'src', 'libSDLPoP.so')
    lib = ctypes.CDLL(so)
    lib.pop_main.argtypes, lib.pop_main.restype = [], None
    lib.play_level_2.argtypes, lib.play_level_2.restype = [], c_int
    lib.init_game.argtypes, lib.init_game.restype = [c_int], None
    lib.rl_get_frame.argtypes = [ctypes.POINTER(ctypes.c_ubyte * (320 * 200 * 3))]
    lib.rl_get_frame.restype = None
    _lib = lib
    return lib


class PoPEnv(gym.Env):

    def __init__(self, visual=False, render_mode=None):
        self.render_mode = render_mode
        self.visual = visual or render_mode == 'human'
        self._orig_cwd = os.getcwd()

        os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SDLPoP"))
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if not self.visual:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_RENDER_DRIVER"] = "software"

        lib = _load_lib()
        c_int.in_dll(lib, "rl_step_mode").value = 1
        c_int.in_dll(lib, "rl_visual_mode").value = int(self.visual or render_mode == "rgb_array")
        c_short.in_dll(lib, "start_level").value = 1
        self.lib = lib

        self.rl_action = c_int.in_dll(lib, "rl_action")
        self.rl_kid_dead = c_int.in_dll(lib, "rl_kid_dead")
        self._snd = c_byte.in_dll(lib, "is_sound_on")
        self._music = c_byte.in_dll(lib, "enable_music")
        self._digi = c_short.in_dll(lib, "digi_playing")
        self._spkr = c_short.in_dll(lib, "speaker_playing")
        self._midi = c_short.in_dll(lib, "midi_playing")
        self._ogg = c_short.in_dll(lib, "ogg_playing")
        self._rseed = c_uint.in_dll(lib, 'random_seed')

        self.kid = (c_uint8 * 16).in_dll(lib, "Kid")
        self.guard = (c_uint8 * 16).in_dll(lib, "Guard")
        self.level = (c_uint8 * 2305).in_dll(lib, "level")
        self.hp_cur = c_short.in_dll(lib, "hitp_curr")
        self.hp_max = c_short.in_dll(lib, "hitp_max")
        self.sword_p = c_short.in_dll(lib, "have_sword")
        self.ghp_cur = c_ushort.in_dll(lib, "guardhp_curr")
        self.ghp_max = c_ushort.in_dll(lib, "guardhp_max")

        self._fbuf = (ctypes.c_ubyte * (320 * 200 * 3))()

        self._grid = np.zeros((N_CH, ROWS, COLS), np.float32)
        self._fstk = np.zeros((FRAME_STACK, N_CH, ROWS, COLS), np.float32)
        self._pstk = np.zeros((FRAME_STACK, PHYS_DIM), np.float32)
        self._ahist = np.zeros(ACT_HIST, np.int64)
        self._nf = 0

        self.observation_space = spaces.Dict({
            "spatial_grid": spaces.Box(-1, 1, (FRAME_STACK * N_CH, ROWS, COLS), np.float32),
            "physics_state": spaces.Box(-1, 1, (FRAME_STACK * PHYS_DIM,), np.float32),
            "action_history": spaces.MultiDiscrete([N_ACTIONS] * ACT_HIST),
        })
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.step_count = 0
        self.current_ep_room_trajectory = []

    def _room_data(self, room):
        lv = np.frombuffer(self.level, dtype=np.uint8)
        off = (room - 1) * 30
        return lv[off:off + 30], lv[720 + off:720 + off + 30]

    def _room_links(self, room):
        lv = np.frombuffer(self.level, dtype=np.uint8)
        b = 1952 + (room - 1) * 4
        return int(lv[b]), int(lv[b + 1]), int(lv[b + 2]), int(lv[b + 3])

    def _fill_ch(self, fg, bg, g):
        t = (fg & 0x1F).reshape(ROWS, COLS)
        m = bg.reshape(ROWS, COLS)

        g[0] = t.astype(np.float32) / 15.0 - 1.0
        g[1] = np.where(t == T_LOOSE, np.where(m != 0, 1.0, 0.0), -1.0)
        g[2] = np.where(t == T_GATE, np.where(m == 0, -1.0, np.where(m >= 188, 1.0, 0.0)), -1.0)
        g[3] = _BTN[t]
        g[4] = np.where(t == T_CHOMPER, (m & 0x7F).astype(np.float32) / 63.5 - 1.0, -1.0)

        item = np.full((ROWS, COLS), -1.0, np.float32)
        pot = (t == T_POTION)
        if pot.any():
            item[pot] = np.array([_POT[int(p)] for p in (m[pot] >> 3) & 7], np.float32)
        item = np.where(t == T_SWORD, 0.71, item)
        item = np.where((t == T_EXIT_LEFT) | (t == T_EXIT_RIGHT), 1.0, item)
        g[5] = item

        g[6] = np.where(t == T_SPIKES, (m & 0x0F).astype(np.float32) / 7.5 - 1.0, -1.0)
        g[7] = np.where((t == T_POTION) & (((m >> 3) & 7) == 3), 1.0, -1.0)

    def _build_grid(self, room):
        g = self._grid
        g[:] = -1.0

        if room < 1 or room > 24:
            g[8], g[9] = _CX, _CY
            return

        fg, bg = self._room_data(room)
        self._fill_ch(fg, bg, g)

        L, R, U, D = self._room_links(room)

        # left neighbor's col 9 feeds our col 0
        if 1 <= L <= 24:
            nfg, nbg = self._room_data(L)
            for r in range(ROWS):
                idx = r * COLS + 9
                bt, bm = int(nfg[idx]) & 0x1F, int(nbg[idx])
                if bt == T_GATE:
                    g[2, r, 0] = max(g[2, r, 0], -1.0 if bm == 0 else (1.0 if bm >= 188 else 0.0))
                if bt == T_CHOMPER:
                    g[4, r, 0] = max(g[4, r, 0], (bm & 0x7F) / 63.5 - 1.0)
                if bt == T_SPIKES:
                    g[6, r, 0] = max(g[6, r, 0], (bm & 0x0F) / 7.5 - 1.0)

        # right neighbor's col 0 feeds our col 9
        if 1 <= R <= 24:
            nfg, nbg = self._room_data(R)
            for r in range(ROWS):
                idx = r * COLS
                bt, bm = int(nfg[idx]) & 0x1F, int(nbg[idx])
                if bt == T_GATE:
                    g[2, r, 9] = max(g[2, r, 9], -1.0 if bm == 0 else (1.0 if bm >= 188 else 0.0))
                if bt == T_CHOMPER:
                    g[4, r, 9] = max(g[4, r, 9], (bm & 0x7F) / 63.5 - 1.0)
                if bt == T_SPIKES:
                    g[6, r, 9] = max(g[6, r, 9], (bm & 0x0F) / 7.5 - 1.0)

        # room above: its row 2 is what we see as our row 0
        if 1 <= U <= 24:
            nfg, nbg = self._room_data(U)
            for c in range(COLS):
                idx = 2 * COLS + c
                bt, bm = int(nfg[idx]) & 0x1F, int(nbg[idx])
                if bt == T_GATE:
                    g[2, 0, c] = max(g[2, 0, c], -1.0 if bm == 0 else (1.0 if bm >= 188 else 0.0))
                if bt == T_CHOMPER:
                    g[4, 0, c] = max(g[4, 0, c], (bm & 0x7F) / 63.5 - 1.0)
                if bt == T_SPIKES:
                    g[6, 0, c] = max(g[6, 0, c], (bm & 0x0F) / 7.5 - 1.0)

        # room below: its row 0 is what we see as our row 2
        if 1 <= D <= 24:
            nfg, nbg = self._room_data(D)
            for c in range(COLS):
                idx = c
                bt, bm = int(nfg[idx]) & 0x1F, int(nbg[idx])
                if bt == T_GATE:
                    g[2, 2, c] = max(g[2, 2, c], -1.0 if bm == 0 else (1.0 if bm >= 188 else 0.0))
                if bt == T_CHOMPER:
                    g[4, 2, c] = max(g[4, 2, c], (bm & 0x7F) / 63.5 - 1.0)
                # bm is the bg/modifier byte SDLPoP pairs with every fg tile id, what it actually
                # holds depends on the tile type: gate openness above, chomper timer just before
                # this, and for spikes it's the low nibble tracking the retract/extend frame
                if bt == T_SPIKES:
                    g[6, 2, c] = max(g[6, 2, c], (bm & 0x0F) / 7.5 - 1.0)

        g[8] = _CX
        g[9] = _CY
        g[10] = (room - 12.5) / 11.5

    def _push(self, action):
        i = self._nf % FRAME_STACK
        k_room = int(self.kid[9])

        self._build_grid(k_room)
        self._fstk[i] = self._grid

        hmax = max(self.hp_max.value, 1)
        g_room = int(self.guard[9])
        g_alive = c_int8(self.guard[13]).value
        gin = (g_room == k_room and self.ghp_max.value > 0 and g_alive == -1)

        p = self._pstk[i]
        p[0] = self.hp_cur.value / hmax
        p[1] = self.hp_max.value / 10.0
        p[2] = np.clip(c_int8(self.kid[8]).value / 10.0, -1, 1)
        p[3] = np.clip(c_int8(self.kid[7]).value / 10.0, -1, 1)
        p[4] = int(self.kid[1]) / 255.0
        p[5] = int(self.kid[2]) / 255.0
        p[6] = 1.0 if c_int8(self.kid[3]).value < 0 else -1.0
        p[7] = 1.0 if self.sword_p.value else -1.0
        p[8] = min(int(self.kid[12]), 2) / 2.0
        p[9] = 1.0 if gin else -1.0
        if gin:
            kc = max(0, min(9, c_int8(self.kid[4]).value))
            gc = max(0, min(9, c_int8(self.guard[4]).value))
            p[10] = (gc - kc) / 9.0
            p[11] = max(0, min(2, c_int8(self.guard[5]).value)) / 2.0
            p[12] = self.ghp_cur.value / max(self.ghp_max.value, 1)
            p[13] = 1.0 if c_int8(self.guard[3]).value < 0 else -1.0
            p[14] = int(self.guard[11]) / 5.0
            p[15] = int(self.guard[6]) / 255.0
        else:
            p[10:16] = -1.0
        # action id + anim frame, added these later so the policy can tell idle vs mid-jump apart
        p[16] = int(self.kid[6]) / 99.0
        p[17] = int(self.kid[0]) / 255.0

        self._ahist[:-1] = self._ahist[1:]
        self._ahist[-1] = action
        self._nf += 1

    def _obs(self):
        n = min(self._nf, FRAME_STACK)

        if self._nf < FRAME_STACK:
            sp = np.zeros((FRAME_STACK * N_CH, ROWS, COLS), np.float32)
            for i in range(n):
                age = n - 1 - i
                idx = (self._nf - 1 - age) % FRAME_STACK
                slot = FRAME_STACK - n + i
                sp[slot * N_CH:(slot + 1) * N_CH] = self._fstk[idx]
        else:
            order = [(self._nf - FRAME_STACK + i) % FRAME_STACK for i in range(FRAME_STACK)]
            sp = self._fstk[order].reshape(FRAME_STACK * N_CH, ROWS, COLS)

        ph = np.zeros(FRAME_STACK * PHYS_DIM, np.float32)
        for i in range(n):
            src = (self._nf - n + i) % FRAME_STACK
            dst = FRAME_STACK - n + i
            ph[dst * PHYS_DIM:(dst + 1) * PHYS_DIM] = self._pstk[src]

        return {
            "spatial_grid": sp,
            "physics_state": ph,
            "action_history": self._ahist.copy(),
        }

    def reset(self, seed=None, options=None):
        global _engine_init
        super().reset(seed=seed)

        if not _engine_init:
            self.lib.pop_main()
            _engine_init = True

        # SDL_AUDIODRIVER=dummy up in __init__ kills actual audio output, but it doesn't touch
        # SDLPoP's own sound-state flags. left alone these carry over between episodes in headless
        # step mode and a stale digi/midi/ogg "playing" flag was enough to crash the engine on
        # some resets, zeroing them here got rid of that
        for h in (self._snd, self._music, self._digi, self._spkr, self._midi, self._ogg):
            h.value = 0

        self.lib.init_game(1)
        # wanted different starts instead of replaying the same seed every episode
        self._rseed.value = seed if seed is not None else int(self.np_random.integers(0, 0xFFFFFFFF))
        self.rl_kid_dead.value = 0

        self._fstk[:] = 0
        self._pstk[:] = 0
        self._ahist[:] = 0
        self._nf = 0
        self.step_count = 0

        self.prev_hp = int(self.hp_cur.value)
        self.prev_ghp = int(self.ghp_cur.value)
        self.prev_room = int(self.kid[9])
        self.has_sword = bool(self.sword_p.value)
        self.g_dead = False
        self.visited = {self.prev_room}

        self.ep_deaths = 0
        self.ep_swords = 0
        self.ep_gkills = 0
        self.ep_levels = 0
        self.g_sword_rooms = set()
        self.current_ep_room_trajectory = [self.prev_room]

        self._push(0)
        return self._obs(), {"room": int(self.kid[9])}

    def step(self, action):
        self.rl_action.value = int(action)
        terminated = False
        reward = 0.0

        for _ in range(FRAME_SKIP):
            self.lib.play_level_2()
            if self.visual:
                time.sleep(1.0 / 15)

            reward -= 0.0025

            hp = int(self.hp_cur.value)
            reward += float(hp - self.prev_hp)
            self.prev_hp = hp

            room = int(self.kid[9])
            ghp = int(self.ghp_cur.value)
            # SDLPoP's leave_guard() zeroes guardhp_curr the moment you leave a room a guard was
            # in, and enter_guard() reloads it to a fresh guardhp_max for whatever guard is in the
            # next room, it's not carried per guard, so without resyncing here that zero-then-
            # refill reads as guard damage, same reason g_dead gets cleared again further down
            if room != self.prev_room:
                self.prev_ghp = ghp
                self.prev_room = room
                self.current_ep_room_trajectory.append(room)

            sword = bool(self.sword_p.value)
            g_room = int(self.guard[9])
            g_alive = c_int8(self.guard[13]).value
            gin = (g_room == room and self.ghp_max.value > 0 and g_alive == -1)

            if sword and gin and room not in self.g_sword_rooms:
                reward += 10.0
                self.g_sword_rooms.add(room)

            dmg = self.prev_ghp - ghp
            if dmg > 0 and sword and not self.g_dead:
                reward += 0.5 * dmg
                if ghp == 0:
                    reward += 3.0
                    self.g_dead = True
                    self.ep_gkills += 1
            self.prev_ghp = ghp

            if sword and not self.has_sword:
                reward += 7.0
                self.has_sword = True
                self.ep_swords += 1
                self.visited = {room}

            if room not in self.visited:
                self.visited.add(room)
                reward += 4.0

            if self.rl_kid_dead.value:
                reward -= 3.0
                terminated = True
                self.ep_deaths += 1
                break

        self.step_count += 1
        self._push(action)

        nxt = c_short.in_dll(self.lib, 'next_level').value
        cur = c_short.in_dll(self.lib, "current_level").value
        if nxt > cur:
            reward += 150.0
            terminated = True
            self.ep_levels += 1

        ghp = int(self.ghp_cur.value)
        if ghp > 0 and self.g_dead:
            self.g_dead = False
            self.prev_ghp = ghp

        info = {
            "room": int(self.kid[9]),
            "hp": int(self.hp_cur.value),
            "step": self.step_count,
            "death": self.ep_deaths,
            "sword_pickup": self.ep_swords,
            "guard_kill": self.ep_gkills,
            "level_completion": self.ep_levels,
            "room_trajectory": self.current_ep_room_trajectory.copy(),
        }
        return self._obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == 'rgb_array':
            self.lib.rl_get_frame(ctypes.byref(self._fbuf))
            return np.frombuffer(self._fbuf, np.uint8).reshape(200, 320, 3).copy()
        return None

    def close(self):
        os.chdir(self._orig_cwd)


def make_env(env_id=0, visual=False, render_mode=None):
    def thunk():
        return gym.wrappers.RecordEpisodeStatistics(PoPEnv(visual=visual, render_mode=render_mode))
    return thunk


def make_vec_env(n_envs, visual=False, render_mode=None):
    return gym.vector.AsyncVectorEnv([make_env(i, visual, render_mode) for i in range(n_envs)], context="spawn")
