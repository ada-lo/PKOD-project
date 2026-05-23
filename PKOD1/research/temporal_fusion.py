import math
import re
from collections import Counter, defaultdict


INDIAN_PLATE_RE = re.compile(
    r"^[A-Z]{2}\s*\d{1,2}\s*[A-Z]{0,3}\s*\d{1,4}$",
    re.IGNORECASE,
)


def normalize_plate_text(text):
    if not text:
        return ""
    normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    return normalized.strip()


def plate_regex_score(text, plate_regex=INDIAN_PLATE_RE):
    if not text:
        return 0.0
    return 1.0 if plate_regex.match(text) else 0.6


def compute_observation_weight(observation, strategy="confidence_quality", plate_regex=INDIAN_PLATE_RE):
    confidence = float(observation.get("ocr_confidence", 0.0) or 0.0)
    quality = float(observation.get("quality_score", 0.0) or 0.0)
    det_conf = float(observation.get("detection_confidence", 0.0) or 0.0)
    regex_boost = plate_regex_score(observation.get("ocr_text", ""), plate_regex=plate_regex)

    if strategy == "confidence_only":
        base = confidence
    elif strategy == "quality_only":
        base = quality
    elif strategy == "detection_confidence":
        base = det_conf
    else:
        base = (0.55 * confidence) + (0.35 * quality) + (0.10 * det_conf)

    return max(1e-6, base * regex_boost)


class TemporalOCRFusion:
    """Independent OCR fusion helper for multi-frame plate recognition."""

    def __init__(self, weighting_strategy="confidence_quality", plate_regex=INDIAN_PLATE_RE):
        self.weighting_strategy = weighting_strategy
        self.plate_regex = plate_regex

    def majority_vote(self, observations):
        texts = [normalize_plate_text(obs.get("ocr_text")) for obs in observations if obs.get("ocr_text")]
        if not texts:
            return ""
        counts = Counter(texts)
        return counts.most_common(1)[0][0]

    def confidence_weighted(self, observations):
        score_by_text = defaultdict(float)
        for obs in observations:
            text = normalize_plate_text(obs.get("ocr_text"))
            if not text:
                continue
            score_by_text[text] += compute_observation_weight(
                obs,
                strategy=self.weighting_strategy,
                plate_regex=self.plate_regex,
            )

        if not score_by_text:
            return ""
        return max(score_by_text.items(), key=lambda item: item[1])[0]

    def character_weighted(self, observations):
        weighted_observations = []
        for obs in observations:
            text = normalize_plate_text(obs.get("ocr_text"))
            if not text:
                continue
            weight = compute_observation_weight(
                obs,
                strategy=self.weighting_strategy,
                plate_regex=self.plate_regex,
            )
            weighted_observations.append((text, weight))

        if not weighted_observations:
            return ""

        weighted_observations.sort(key=lambda item: item[1], reverse=True)
        target_len = round(
            sum(len(text) * weight for text, weight in weighted_observations)
            / sum(weight for _, weight in weighted_observations)
        )
        target_len = max(1, int(target_len))

        fused_chars = []
        for idx in range(target_len):
            char_scores = defaultdict(float)
            for text, weight in weighted_observations:
                if idx < len(text):
                    char_scores[text[idx]] += weight
                else:
                    char_scores[""] += weight * 0.35
            best_char = max(char_scores.items(), key=lambda item: item[1])[0]
            if best_char:
                fused_chars.append(best_char)

        fused = "".join(fused_chars)
        if self.plate_regex.match(fused):
            return fused

        best_candidate = max(
            weighted_observations,
            key=lambda item: (plate_regex_score(item[0], self.plate_regex), item[1]),
        )[0]
        if plate_regex_score(best_candidate, self.plate_regex) >= plate_regex_score(fused, self.plate_regex):
            return best_candidate
        return fused

    def run_all(self, observations):
        return {
            "majority": self.majority_vote(observations),
            "confidence_weighted": self.confidence_weighted(observations),
            "character_weighted": self.character_weighted(observations),
        }
