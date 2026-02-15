import numpy as np
import config
from collections import defaultdict
class VehicleEventManager:
    """State-machine based event manager implementing:
    DETECTED -> APPROACHING -> ARMED -> CROSSED -> CONFIRMED -> FINALIZED
    """
    def __init__(self):
        self.state = {}  # track_id -> state dict
        self.counted = defaultdict(set)  # track_id -> set('entry'|'exit')

    def _side(self, y):
        return 'above' if y < config.LINE[1] else 'below'

    def _movement_vector(self, hist):
        # return approximate dy over recent history (negative = moving up)
        if len(hist) < 2:
            return 0
        ys = [p[1] for p in hist]
        return ys[-1] - ys[0]

    def process(self, track_id, center_y, pos_history, frame_idx):
        """Process one observation; return 'entry'/'exit' when FINALIZED else None."""
        st = self.state.setdefault(track_id, {
            'state': 'DETECTED',
            'first_seen': frame_idx,
            'last_seen': frame_idx,
            'approach_dir': None,   # 'up' or 'down'
            'approach_frames': 0,
            'arm_frames': 0,
            'cross_frame': None,
            'confirmed_frames': 0,
            'finalized': set(),
        })

        st['last_seen'] = frame_idx

        # If already finalized for both directions, skip
        if 'entry' in st['finalized'] and 'exit' in st['finalized']:
            return None

        # If the id already had this direction finalized, ensure we don't re-count
        # (Handled at finalization)

        # Ensure we have enough history
        hist = pos_history[-config.DIRECTION_STABLE_FRAMES:] if pos_history else []

        # DETECTED -> require several frames and a reasonably stable movement
        if st['state'] == 'DETECTED':
            if len(pos_history) >= config.DETECTED_MIN_FRAMES:
                mv = self._movement_vector(pos_history[-config.DETECTED_MIN_FRAMES:])
                # movement magnitude threshold (small value to allow slow movement)
                if abs(mv) >= 2:
                    # determine approach direction toward the line
                    # if moving up (dy negative) and currently below line => approaching up
                    if mv < 0 and np.median([p[1] for p in pos_history[-config.DETECTED_MIN_FRAMES:]]) > config.LINE[1]:
                        st['approach_dir'] = 'up'
                        st['state'] = 'APPROACHING'
                    elif mv > 0 and np.median([p[1] for p in pos_history[-config.DETECTED_MIN_FRAMES:]]) < config.LINE[1]:
                        st['approach_dir'] = 'down'
                        st['state'] = 'APPROACHING'
            return None

        # APPROACHING -> monitor distance decreasing and consistent direction
        if st['state'] == 'APPROACHING':
            # compute if distance to line is decreasing over short window
            if len(pos_history) >= 2:
                dists = [abs(p[1] - config.LINE[1]) for p in pos_history[-config.DIRECTION_STABLE_FRAMES:]]
                if len(dists) >= 2 and dists[-1] < dists[0]:
                    st['approach_frames'] += 1
                else:
                    st['approach_frames'] = max(0, st['approach_frames'] - 1)

            # if near the line consistently, move to ARMED
            if st['approach_frames'] >= config.ARM_MIN_FRAMES and abs(center_y - config.LINE[1]) <= config.PRE_ZONE_PIXELS:
                st['state'] = 'ARMED'
                st['arm_frames'] = 0
            return None

        # ARMED -> wait for a monotonic crossing matching approach direction
        if st['state'] == 'ARMED':
            st['arm_frames'] += 1
            # reset if direction reversed
            if st['approach_dir'] == 'up' and center_y < config.LINE[1] and np.median([p[1] for p in hist]) > config.LINE[1]:
                # crossing candidate (below -> above)
                st['state'] = 'CROSSED'
                st['cross_frame'] = frame_idx
                return None
            if st['approach_dir'] == 'down' and center_y > config.LINE[1] and np.median([p[1] for p in hist]) < config.LINE[1]:
                st['state'] = 'CROSSED'
                st['cross_frame'] = frame_idx
                return None

            # allow staying armed if still near the line; if moves away, drop back to APPROACHING
            if abs(center_y - config.LINE[1]) > config.PRE_ZONE_PIXELS * 1.5:
                st['state'] = 'APPROACHING'
                st['approach_frames'] = 0
            return None

        # CROSSED -> require displacement and persistence to CONFIRMED
        if st['state'] == 'CROSSED':
            # determine intended direction
            if st['approach_dir'] == 'up':
                intended = 'entry'
                # need to be further above the line
                delta = config.LINE[1] - center_y
            else:
                intended = 'exit'
                delta = center_y - config.LINE[1]

            if delta >= config.MIN_DISPLACEMENT:
                st['confirmed_frames'] += 1
            else:
                st['confirmed_frames'] = 0

            # require monotonic movement: check last few positions stay on the same side
            side_flags = [1 if p[1] < config.LINE[1] else 0 for p in pos_history[-config.CONFIRM_FRAMES:]]
            side_consistent = all(x == side_flags[0] for x in side_flags) if len(side_flags) >= config.CONFIRM_FRAMES else False

            if st['confirmed_frames'] >= config.CONFIRM_FRAMES and side_consistent:
                # FINALIZE
                st['state'] = 'FINALIZED'
                st['finalized'].add(intended)
                # set cooldown implicitly by remembering finalized direction
                return intended

            # If vehicle reverses back across the line, degrade to ARMED
            if st['approach_dir'] == 'up' and center_y > config.LINE[1]:
                st['state'] = 'ARMED'
                st['confirmed_frames'] = 0
            if st['approach_dir'] == 'down' and center_y < config.LINE[1]:
                st['state'] = 'ARMED'
                st['confirmed_frames'] = 0

            return None

        # FINALIZED: shouldn't get here often; ignore further events for this direction
        return None