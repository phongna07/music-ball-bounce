"""Stage 2: turn an analysis into a ball path and a set of plates.

The ball is never simulated forward. Between two impacts it follows a closed-form
parabola, so for each note we choose where the ball should be at that instant,
put a plate there, and tilt the plate so the reflection produces the velocity the
next arc needs. A depth-first search with backtracking picks each step so that no
arc clips a plate it should not touch. `verify` then re-checks every segment with
a fine time step.
"""

from __future__ import annotations

import math
import random

import numpy as np
from scipy.ndimage import gaussian_filter1d

BALL_R = 0.35
PLATE_TH = 0.14
CONTACT = BALL_R + PLATE_TH / 2  # ball centre to plate centre line at impact
CORRIDOR = 3.1  # half width the path should stay inside
CAM_FPS = 60
VIDEO_FPS = 60
FINE_STEP = 0.0005  # time step for arcs that touch the plate being placed (matches verify)
SELF_TOL = 1e-3  # allowed dip below CONTACT for those arcs; verify allows 2e-3

# two balls: each gets a lane; nothing of one lane may cross x = 0 (+/- LANE_GAP)
LANE_CX = 2.15  # lane centre lines at -/+ this
LANE_CORRIDOR = 1.15  # soft half width of a lane
LANE_GAP = 0.12  # clearance either side of the centre line
LANE_OUTER = 4.3  # hard limit for the ball centre on the outer side
DRIFT_MAX = 2.6  # how far an impact may sit from the shared descent line
DRIFT_COST = 0.25
DUO_GAIN = 1.3  # auto mode: two balls once they catch 30 % more of the music
DUO_CAMERA_DIST = 29.0  # pulled back from 25.5 so both lanes fit

PALETTES = {
    "neon": ["#2f8bff", "#ffd60a", "#3ddc84", "#b388ff", "#ff4d5a", "#ff9f1c", "#ff8fd0", "#3df0e0"],
    "sunset": ["#ff5e5b", "#ffb347", "#ffd97d", "#ff8fab", "#c77dff", "#7b2cbf", "#ff9770", "#f72585"],
    "ice": ["#48cae4", "#90e0ef", "#ade8f4", "#7b9cff", "#b8c0ff", "#64dfdf", "#e0fbfc", "#5390d9"],
}


# --------------------------------------------------------------------------
# note selection
# --------------------------------------------------------------------------
def _pick(analysis: dict, gap: float) -> list[dict]:
    """Strongest onsets first, keeping only those at least `gap` from every one kept."""
    dur = analysis["duration"]
    cands = [n for n in analysis["notes"] if n["strength"] >= 0.1 and 0.25 <= n["t"] <= dur - 0.2]
    cands.sort(key=lambda n: -(n["strength"] * (0.7 + 0.6 * n["melodic"])))
    chosen: list[dict] = []
    for n in cands:
        if all(abs(n["t"] - c["t"]) >= gap for c in chosen):
            chosen.append(dict(n))
    chosen.sort(key=lambda n: n["t"])
    return chosen


def _fill(chosen: list[dict], analysis: dict, min_gap: float, max_gap: float) -> list[dict]:
    """Fill long silences from the beat grid so the ball never hangs."""
    dur = analysis["duration"]
    beats = analysis["beats"]
    filled = True
    while filled:
        filled = False
        times = [0.0] + [c["t"] for c in chosen] + [dur]
        for a, b in zip(times[:-1], times[1:]):
            if b - a <= max_gap:
                continue
            inside = [t for t in beats if a + min_gap <= t <= b - min_gap and 0.25 <= t <= dur - 0.2]
            if inside:
                mid = min(inside, key=lambda t: abs(t - (a + b) / 2))
                chosen.append({"t": mid, "strength": 0.3, "melodic": 0.5, "pitch": None})
                chosen.sort(key=lambda n: n["t"])
                filled = True
                break
    return chosen


def _shift(chosen: list[dict], offset: float, snap_fps: int | None = None) -> list[dict]:
    out: list[dict] = []
    prev_frame = -1
    for c in chosen:
        t = float(c["t"]) + offset
        if t <= 0.05:
            continue
        item = dict(c)
        if snap_fps:
            frame = max(prev_frame + 1, int(round(t * snap_fps)))
            prev_frame = frame
            item["t"] = round(frame / snap_fps, 6)
        else:
            item["t"] = round(t, 6)
        out.append(item)
    return out


def select_notes(
    analysis: dict,
    min_gap: float,
    max_gap: float,
    offset: float,
    snap_fps: int | None = VIDEO_FPS,
) -> list[dict]:
    return _shift(_fill(_pick(analysis, min_gap), analysis, min_gap, max_gap), offset, snap_fps)


def _weight(notes: list[dict]) -> float:
    return sum(n["strength"] for n in notes)


def split_notes(analysis: dict, min_gap: float) -> tuple[list[list[dict]], float]:
    """Pick notes for two balls and deal them out, one lane per ball.

    The combined stream may be twice as dense as one ball allows; each lane still
    keeps `min_gap` between its own bounces. A note goes to whichever ball has been
    waiting longer, nudged by pitch (low notes left, high notes right) so the
    melody reads across the screen. Returns the lanes (unfilled, unshifted) and
    how much more of the music (strength-weighted) two balls catch than one.
    """
    dense = _pick(analysis, min_gap / 2)
    pitches = [n["pitch"] for n in dense if n["pitch"] is not None]
    split = float(np.median(pitches)) if pitches else 0.0
    lanes: list[list[dict]] = [[], []]
    last = [-1e9, -1e9]
    for n in dense:
        ok = [k for k in (0, 1) if n["t"] - last[k] >= min_gap]
        if not ok:
            continue
        def score(k):
            idle = n["t"] - last[k]
            want = None if n["pitch"] is None else (0 if n["pitch"] < split else 1)
            return min(idle, 2.0) + (0.12 if want == k else 0.0)
        k = max(ok, key=score)
        lanes[k].append(n)
        last[k] = n["t"]
    single = _weight(_pick(analysis, min_gap))
    gain = (_weight(lanes[0]) + _weight(lanes[1])) / max(single, 1e-9)
    return lanes, gain


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _pt_seg_dist(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distances from points (M,2) to segments a->b (K,2). Returns (M,K)."""
    ab = b - a
    ap = pts[:, None, :] - a[None, :, :]
    denom = np.maximum((ab * ab).sum(-1), 1e-12)
    u = np.clip((ap * ab[None]).sum(-1) / denom[None], 0.0, 1.0)
    closest = a[None] + u[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - closest, axis=-1)


def _arc_points(p, v, g, dt, step=0.02) -> tuple[np.ndarray, np.ndarray]:
    n = max(8, int(dt / step) + 1)
    tau = np.linspace(0.0, dt, n)
    pts = np.stack([p[0] + v[0] * tau, p[1] + v[1] * tau - 0.5 * g * tau * tau], axis=1)
    return tau, pts


def _plate_segment(p, n, length, off):
    tx, ty = n[1], -n[0]
    cx = p[0] - n[0] * CONTACT + tx * off
    cy = p[1] - n[1] * CONTACT + ty * off
    h = length / 2
    return (cx, cy), np.array([cx - tx * h, cy - ty * h]), np.array([cx + tx * h, cy + ty * h])


# --------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------
class Stuck(Exception):
    def __init__(self, index: int):
        self.index = index


SINGLE_LANE = {"cx": 0.0, "corridor": CORRIDOR}


def _solve(times: list[float], g: float, rng: random.Random, t_end: float,
           lane: dict = SINGLE_LANE) -> tuple[list[dict], float]:
    """Solve one ball's path. `lane` gives its centre line and soft corridor; a lane
    may also carry hard x limits for the ball ("ball_x") and for the plates'
    outline ("plate_x"), and a shared descent `y_ref(t)` the impacts must follow."""
    n = len(times)
    lead = min(max(times[0], 0.5), 1.1)
    cx, corridor = lane["cx"], lane["corridor"]
    ball_lo, ball_hi = lane.get("ball_x", (-math.inf, math.inf))
    plate_lo, plate_hi = lane.get("plate_x", (-math.inf, math.inf))
    y_ref = lane.get("y_ref")
    y0 = y_ref(times[0]) if y_ref else 0.0
    # state before impact i
    pos = [np.array([cx, y0])]
    vin = [np.array([0.0, -g * lead])]
    steps: list[dict] = []  # one per impact: plate + outgoing arc
    seg_a: list[np.ndarray] = []
    seg_b: list[np.ndarray] = []
    arcs: list[np.ndarray] = []
    # the lead-in fall is an arc too
    tau0 = np.linspace(0, lead, 24)
    arcs.append(np.stack([np.full_like(tau0, cx), y0 + 0.5 * g * (lead - tau0) ** 2], axis=1)[:-2])

    iters: dict[int, list] = {}
    backtracks = 0
    deepest = 0
    WINDOW = 200  # same neighbourhood verify checks

    def candidates(i: int) -> list:
        p, v_in = pos[i], vin[i]
        dt = (times[i + 1] - times[i]) if i + 1 < n else min(1.4, max(0.6, t_end - times[i]))
        prev_dt = times[i] - times[i - 1] if i > 0 else lead
        length = float(np.clip(0.85 + 1.3 * min(dt, prev_dt), 0.9, 1.7))
        dy_target = -min(0.9 + 1.6 * dt, 4.0)
        speed_in = float(np.linalg.norm(v_in))
        out = []
        for dx0 in np.linspace(-3.4, 3.4, 15):
            for dy0 in (-4.2, -3.2, -2.4, -1.7, -1.1, -0.6, -0.1, 0.5):
                dx = dx0 + rng.uniform(-0.2, 0.2)
                dy = dy0 + rng.uniform(-0.2, 0.2)
                v_out = np.array([dx / dt, dy / dt + 0.5 * g * dt])
                d = v_out - v_in
                dn = float(np.linalg.norm(d))
                if dn < 1e-6:
                    continue
                nrm = d / dn
                vn_in, vn_out = float(v_in @ nrm), float(v_out @ nrm)
                speed_out = float(np.linalg.norm(v_out))
                if vn_in >= 0 or vn_out <= 0 or nrm[1] < -0.15:
                    continue
                e = -vn_out / vn_in
                if not 0.3 <= e <= 1.6:
                    continue
                if -vn_in / speed_in < 0.3 or vn_out / max(speed_out, 1e-6) < 0.25:
                    continue
                x1 = p[0] + dx
                rel = x1 - cx
                if abs(rel) > corridor + 1.0 or not ball_lo <= x1 <= ball_hi:
                    continue
                drift = 0.0
                if y_ref:
                    drift = p[1] + dy - y_ref(times[i] + dt)
                    if abs(drift) > DRIFT_MAX:
                        continue
                rise = v_out[1] ** 2 / (2 * g) if v_out[1] > 0 else 0.0
                cost = 0.18 * (dy - dy_target) ** 2
                cost += 0.5 * max(0.0, rise - 3.0) ** 2
                cost += 3.0 * max(0.0, abs(rel) - corridor) ** 2
                cost += 0.04 * rel * rel
                cost += DRIFT_COST * drift * drift
                cost += 4.0 * max(0.0, 0.6 - e) ** 2 + 4.0 * max(0.0, e - 1.15) ** 2
                cost += 1.2 * max(0.0, 0.45 - nrm[1])
                cost += 0.6 * max(0.0, 1.0 - abs(dx))  # avoid dull vertical ping-pong
                cost += rng.uniform(0, 0.6)
                off = rng.uniform(-0.25, 0.25) * length
                out.append((cost, dt, v_out, nrm, e, length, off))
        out.sort(key=lambda c: c[0])
        return out

    def valid(i: int, cand) -> dict | None:
        _, dt, v_out, nrm, e, length, off = cand
        p = pos[i]
        centre, a, b = _plate_segment(p, nrm, length, off)
        if min(a[0], b[0]) - PLATE_TH < plate_lo or max(a[0], b[0]) + PLATE_TH > plate_hi:
            return None
        lo = max(0, len(seg_a) - WINDOW)
        # new plate against existing plates
        if len(seg_a) > lo:
            A, B = np.array(seg_a[lo:]), np.array(seg_b[lo:])
            samples = a[None] + np.linspace(0, 1, 9)[:, None] * (b - a)[None]
            if _pt_seg_dist(samples, A, B).min() < PLATE_TH + 0.4:
                return None
        # new plate against earlier arcs (the arriving arc may touch it at the end)
        alo = max(0, len(arcs) - WINDOW)
        if len(arcs) - 1 > alo:
            old = np.concatenate(arcs[alo:-1])
            if _pt_seg_dist(old, a[None], b[None]).min() < CONTACT + 0.1:
                return None
        # The arcs touching this plate (arriving and outgoing) are checked at the same
        # fine resolution and tighter tolerance than `verify`, otherwise a shallow
        # graze slips through the coarse samples and verification rejects the path.
        if i > 0:
            prev = steps[i - 1]
            ftau, fpts = _arc_points(pos[i - 1], prev["v_out"], g, prev["dt"], step=FINE_STEP)
            arriving = fpts[(ftau > 0.002) & (ftau < prev["dt"] - 0.002)]
        else:
            arriving = arcs[-1]
        if arriving.size and _pt_seg_dist(arriving, a[None], b[None]).min() < CONTACT - SELF_TOL:
            return None
        ftau, fpts = _arc_points(p, v_out, g, dt, step=FINE_STEP)
        keep = fpts[ftau > 0.002]
        if _pt_seg_dist(keep, a[None], b[None]).min() < CONTACT - SELF_TOL:
            return None
        # outgoing arc against every other plate
        tau, pts = _arc_points(p, v_out, g, dt)
        if len(seg_a) > lo and _pt_seg_dist(pts, A, B).min() < CONTACT + 0.1:
            return None
        # stored arc excludes the last instants, which belong to the next plate
        return {
            "dt": dt, "v_out": v_out, "n": nrm, "e": e, "len": length, "off": off,
            "centre": centre, "a": a, "b": b, "arc": pts[tau < dt - 0.03],
        }

    i = 0
    while i < n:
        deepest = max(deepest, i)
        if i not in iters:
            iters[i] = candidates(i)
        placed = None
        while iters[i]:
            placed = valid(i, iters[i].pop(0))
            if placed:
                break
        if placed:
            steps.append(placed)
            seg_a.append(placed["a"])
            seg_b.append(placed["b"])
            arcs.append(placed["arc"])
            dt, v_out = placed["dt"], placed["v_out"]
            pos.append(pos[i] + v_out * dt + np.array([0.0, -0.5 * g * dt * dt]))
            vin.append(v_out + np.array([0.0, -g * dt]))
            i += 1
        else:
            backtracks += 1
            del iters[i]
            if i == 0 or backtracks > 4000:
                raise Stuck(deepest)
            i -= 1
            steps.pop(); seg_a.pop(); seg_b.pop(); arcs.pop(); pos.pop(); vin.pop()

    for k, s in enumerate(steps):
        s["t"], s["p"], s["v_in"] = times[k], pos[k], vin[k]
    return steps, lead


# --------------------------------------------------------------------------
# verification: independent fine-step check of every segment
# --------------------------------------------------------------------------
def _ball_xy(segs, t_arr, g):
    starts = np.array([s["t0"] for s in segs])
    idx = np.clip(np.searchsorted(starts, t_arr, side="right") - 1, 0, len(segs) - 1)
    s0 = np.array([[segs[j][k] for k in ("t0", "x", "y", "vx", "vy")] for j in idx])
    tau = t_arr - s0[:, 0]
    return np.stack([s0[:, 1] + s0[:, 3] * tau, s0[:, 2] + s0[:, 4] * tau - 0.5 * g * tau * tau], axis=1)


def verify(lanes: list[list[dict]], g: float, ball_segs: list[list[dict]] | None = None) -> dict:
    """Check every ball against its own plates and against every other ball's plates;
    with two balls, also check the balls never touch each other."""
    worst_gap, worst_refl = 0.0, 0.0
    for li, steps in enumerate(lanes):
        A = np.array([s["a"] for s in steps])
        B = np.array([s["b"] for s in steps])
        others = [s for lj, o in enumerate(lanes) if lj != li for s in o]
        OA = np.array([s["a"] for s in others]).reshape(-1, 2)
        OB = np.array([s["b"] for s in others]).reshape(-1, 2)
        tag = f"ball {li} " if len(lanes) > 1 else ""
        for i, s in enumerate(steps):
            # reflection law holds at this plate
            vn = s["v_in"] @ s["n"]
            expect = s["v_in"] - (1 + s["e"]) * vn * s["n"]
            worst_refl = max(worst_refl, float(np.linalg.norm(expect - s["v_out"])))
            # contact is on the flat face, at exactly the contact distance
            ab = s["b"] - s["a"]
            u = float((s["p"] - s["a"]) @ ab / (ab @ ab))
            d = float(_pt_seg_dist(s["p"][None], s["a"][None], s["b"][None])[0, 0])
            if not (0.05 < u < 0.95):
                raise AssertionError(f"{tag}note {i}: ball lands on the plate edge (u={u:.3f})")
            worst_gap = max(worst_gap, abs(d - CONTACT))
            # nothing is penetrated on the way to the next impact
            tau, pts = _arc_points(s["p"], s["v_out"], g, s["dt"], step=0.0005)
            inner = pts[(tau > 0.002) & (tau < s["dt"] - 0.002)]
            lo, hi = max(0, i - 200), min(len(steps), i + 200)
            dist = _pt_seg_dist(inner, A[lo:hi], B[lo:hi])
            if dist.min() < CONTACT - 2e-3:
                j = lo + int(np.unravel_index(np.argmin(dist), dist.shape)[1])
                raise AssertionError(f"{tag}arc after note {i} penetrates plate {j} ({dist.min():.4f})")
            if len(OA):
                # the other ball's plates must stay clear of this ball entirely
                dist = _pt_seg_dist(pts, OA, OB)
                if dist.min() < CONTACT + 0.05:
                    raise AssertionError(f"{tag}arc after note {i} touches the other ball's plates")
    if worst_gap > 1e-6 or worst_refl > 1e-6:
        raise AssertionError(f"contact error {worst_gap:.2e}, reflection error {worst_refl:.2e}")
    report = {"segments": sum(len(s) for s in lanes), "contact_error": worst_gap, "reflection_error": worst_refl}
    if ball_segs and len(ball_segs) > 1:
        t0 = min(s[0]["t0"] for s in ball_segs)
        t1 = max(st[-1]["t"] + st[-1]["dt"] for st in lanes)
        t_arr = np.arange(t0, t1, 1.0 / 480)
        paths = [_ball_xy(segs, np.maximum(t_arr, segs[0]["t0"]), g) for segs in ball_segs]
        gap = float(np.linalg.norm(paths[0] - paths[1], axis=1).min())
        if gap < 2 * BALL_R + 0.1:
            raise AssertionError(f"the balls come within {gap:.3f} of each other")
        report["ball_clearance"] = round(gap, 3)
    return report


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def _solve_lane(notes: list[dict], g: float, seed: int, duration: float, lane: dict) -> tuple[list[dict], float, list[dict], int]:
    """Solve with retries; after three failures on the same notes, drop the one it chokes on."""
    notes = list(notes)
    dropped = 0
    attempt = 0
    while True:
        try:
            steps, lead = _solve([n["t"] for n in notes], g, random.Random(seed + attempt), duration, lane)
            return steps, lead, notes, dropped
        except Stuck as s:
            attempt += 1
            if attempt % 3 == 0:
                notes.pop(min(s.index + 1, len(notes) - 1))
                dropped += 1
            if len(notes) < 2 or attempt > 200:
                raise SystemExit("Could not find a collision-free path for this track.")


def _descent_rate(lanes: list[list[dict]]) -> float:
    """Average fall speed a single ball would settle into on these notes (matches
    the dy_target the solver aims for), so two lanes can share one descent line."""
    rates = []
    for notes in lanes:
        t = np.array([n["t"] for n in notes])
        drop = np.minimum(0.9 + 1.6 * np.diff(t), 4.0).sum()
        rates.append(drop / max(t[-1] - t[0], 1e-6))
    return float(np.mean(rates))


def choreograph(analysis: dict, seed: int = 1, palette: str = "neon", min_gap: float = 0.17,
                max_gap: float = 1.8, offset: float = 0.0, balls: str | int = "auto",
                snap_fps: int | None = VIDEO_FPS) -> dict:
    duration = analysis["duration"]
    duo_notes, gain = split_notes(analysis, min_gap)
    if balls == "auto":
        count = 2 if gain >= DUO_GAIN else 1
    else:
        count = int(balls)
        if count not in (1, 2):
            raise SystemExit("--balls must be auto, 1 or 2")

    if count == 1:
        lane_notes = [select_notes(analysis, min_gap, max_gap, offset, snap_fps=snap_fps)]
    else:
        duo_max_gap = min(max_gap, 1.35)
        lane_notes = [_shift(_fill(notes, analysis, min_gap, duo_max_gap), offset, snap_fps) for notes in duo_notes]
    if any(len(n) < 2 for n in lane_notes):
        raise SystemExit("Not enough onsets found in this audio to choreograph.")
    iois = np.concatenate([np.diff([n["t"] for n in notes]) for notes in lane_notes])
    g = float(np.clip(4.8 / float(np.median(iois)) ** 2, 14.0, 55.0))

    if count == 1:
        lanes = [SINGLE_LANE]
    else:
        rate = _descent_rate(lane_notes)
        t_first = min(notes[0]["t"] for notes in lane_notes)

        def y_ref(t: float) -> float:
            return -rate * (t - t_first)

        inner = LANE_GAP + BALL_R
        lanes = [
            {"cx": -LANE_CX, "corridor": LANE_CORRIDOR, "ball_x": (-LANE_OUTER, -inner),
             "plate_x": (-math.inf, -LANE_GAP), "y_ref": y_ref},
            {"cx": LANE_CX, "corridor": LANE_CORRIDOR, "ball_x": (inner, LANE_OUTER),
             "plate_x": (LANE_GAP, math.inf), "y_ref": y_ref},
        ]

    solved = []
    dropped = 0
    for k, (notes, lane) in enumerate(zip(lane_notes, lanes)):
        # lane 0 keeps the single-ball seed sequence, so one-ball layouts are unchanged
        steps, lead, notes, d = _solve_lane(notes, g, seed * 1000 + k * 100_000, duration, lane)
        solved.append((steps, lead, notes, lane))
        dropped += d

    colours = PALETTES[palette]
    sections = analysis["sections"]

    def section_of(t):
        for k, s in enumerate(sections):
            if s["t0"] <= t < s["t1"]:
                return k
        return len(sections) - 1

    plates: list[dict] = []
    ball_segs: list[list[dict]] = []
    for ball, (steps, lead, notes, lane) in enumerate(solved):
        last_c = -1
        for k, (s, note) in enumerate(zip(steps, notes)):
            if note["pitch"] is not None:
                c = ((note["pitch"] % 12) * 7 % 12) * len(colours) // 12
            else:
                c = k
            c = (c + 3 * section_of(s["t"])) % len(colours)
            if c == last_c:
                c = (c + 1) % len(colours)
            last_c = c
            plate_entry = {
                "t": round(s["t"], 6),
                "x": round(s["centre"][0], 4), "y": round(s["centre"][1], 4),
                "nx": round(float(s["n"][0]), 5), "ny": round(float(s["n"][1]), 5),
                "len": round(s["len"], 3),
                "hx": round(float(s["p"][0] - s["n"][0] * BALL_R), 4),
                "hy": round(float(s["p"][1] - s["n"][1] * BALL_R), 4),
                "color": colours[c],
                "strength": note["strength"],
                "speed": round(float(abs(s["v_in"] @ s["n"])), 3),
                "ball": ball,
                "_k": k,
            }
            if snap_fps:
                plate_entry["frame"] = int(round(s["t"] * snap_fps))
            plates.append(plate_entry)
        p0 = steps[0]["p"]
        segs = [{"t0": steps[0]["t"] - lead, "x": float(p0[0]), "y": float(p0[1]) + 0.5 * g * lead * lead,
                 "vx": 0.0, "vy": 0.0}]
        for s in steps:
            segs.append({"t0": s["t"], "x": float(s["p"][0]), "y": float(s["p"][1]),
                         "vx": float(s["v_out"][0]), "vy": float(s["v_out"][1])})
        ball_segs.append(segs)

    report = verify([s[0] for s in solved], g, ball_segs)
    report["dropped_notes"] = dropped
    report["balls"] = count
    report["duo_gain"] = round(gain, 3)
    report["offset"] = offset
    report["snap_fps"] = snap_fps
    report["note_source"] = analysis.get("note_source", "audio")

    # one plate list for the whole scene, in hit order; each ball maps its own hits into it
    plates.sort(key=lambda p: (p["t"], p["ball"]))
    index = [[0] * len(s[0]) for s in solved]
    for gi, p in enumerate(plates):
        index[p["ball"]][p.pop("_k")] = gi

    # camera: zero-phase smoothed (mean) ball path, clamped so it never runs ahead of the fall
    t_arr = np.arange(0, duration + 1.0, 1.0 / CAM_FPS)
    paths = []
    for segs, (steps, *_rest) in zip(ball_segs, solved):
        xy = _ball_xy(segs, np.maximum(t_arr, segs[0]["t0"]), g)
        late = t_arr > steps[-1]["t"] + steps[-1]["dt"]
        if late.any():
            xy[late] = xy[~late][-1]
        paths.append(xy)
    xy = np.mean(paths, axis=0)
    cam_x = gaussian_filter1d(xy[:, 0], 0.7 * CAM_FPS, mode="nearest") * 0.55
    cam_y = gaussian_filter1d(xy[:, 1], 0.4 * CAM_FPS, mode="nearest")

    for segs in ball_segs:
        for s in segs:
            for key in s:
                s[key] = round(s[key], 6 if key == "t0" else 5)

    return {
        "version": 2,
        "duration": duration,
        "tempo": analysis["tempo"],
        "gravity": round(g, 4),
        "ballRadius": BALL_R,
        "plateThickness": PLATE_TH,
        "balls": [{"segments": segs, "plates": idx} for segs, idx in zip(ball_segs, index)],
        "plates": plates,
        "camera": {"fps": CAM_FPS, "x": np.round(cam_x, 3).tolist(), "y": np.round(cam_y, 3).tolist(),
                   "dist": DUO_CAMERA_DIST if count > 1 else 25.5},
        "kicks": analysis["kicks"],
        "beats": analysis["beats"],
        "energy": analysis["energy"],
        "sections": sections,
        "report": report,
    }
