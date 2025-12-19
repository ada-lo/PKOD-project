import math
import time
import config

class TrackletBuffer:
    """Keeps recently lost tracklets for time-based re-association.
    Stores: positions, features, last_time.
    """
    def __init__(self, keep_secs=config.LOST_BUFFER_SECS, max_dist=config.MAX_REASSOC_DISTANCE):
        self.keep_secs = keep_secs
        self.max_dist = max_dist
        self.buffer = {}  # old_id -> {'positions':[], 'features':[], 'last_time':float}

    def add_lost(self, old_id, features, positions, last_time):
        self.buffer[old_id] = {
            'features': list(features) if features is not None else [],
            'positions': list(positions) if positions is not None else [],
            'last_time': float(last_time),
        }

    def prune(self, now=None):
        if now is None:
            now = time.time()
        to_del = [k for k, v in self.buffer.items() if now - v['last_time'] > self.keep_secs]
        for k in to_del:
            del self.buffer[k]

    def reassociate(self, start_pos, now=None):
        """Return best-matching old_id or None.
        Match by motion/spatial first, then appearance as tie-breaker.
        start_pos: (x,y)
        """
        if now is None:
            now = time.time()

        best_id = None
        best_score = 0.0
        scores = []

        for old_id, info in list(self.buffer.items()):
            # skip expired
            if now - info['last_time'] > self.keep_secs:
                continue

            # spatial distance from last known position
            if info['positions']:
                last_pos = info['positions'][-1]
                dist = math.hypot(start_pos[0] - last_pos[0], start_pos[1] - last_pos[1])
            else:
                dist = float('inf')

            if dist > self.max_dist:
                continue

            # motion continuity: estimate speed if possible (pixels per frame assumed)
            motion_score = 1.0 - (dist / (self.max_dist + 1e-6))

            # appearance score: compare new placeholder (we don't have features for new start)
            # use stored features as tie-breaker with small weight
            appearance_score = 0.0
            if info['features']:
                # compare last two features as representative
                f = info['features'][-1]
                # cannot compute similarity without a current feature, so give neutral 0.5
                appearance_score = 0.5

            score = (1.0 - config.APPEARANCE_WEIGHT) * motion_score + config.APPEARANCE_WEIGHT * appearance_score
            scores.append((old_id, score, dist))

        if not scores:
            return None

        # pick best score but guard against ambiguous ties
        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score, best_dist = scores[0]
        if len(scores) > 1 and scores[1][1] >= best_score * 0.85:
            # ambiguous: multiple good matches — avoid unsafe reassociation
            return None

        # Accept only if score positive
        if best_score > 0:
            # remove from buffer to avoid double-association
            if best_id in self.buffer:
                del self.buffer[best_id]
            return best_id

        return None