import math

import cv2 as cv
import numpy as np


def _clip01(value):
    return max(0.0, min(1.0, float(value)))


def _normalize_blur_score(laplacian_var, target=180.0):
    return _clip01(laplacian_var / float(target))


def _normalize_brightness_score(mean_intensity, target=145.0, tolerance=110.0):
    distance = abs(float(mean_intensity) - target)
    return _clip01(1.0 - (distance / float(tolerance)))


def _normalize_contrast_score(std_dev, target=70.0):
    return _clip01(float(std_dev) / float(target))


def compute_frame_quality(image, ocr_confidence=0.0, weights=None):
    """Return normalized frame quality metrics for an OCR crop."""
    if image is None or image.size == 0:
        return {
            "quality_score": 0.0,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "contrast_score": 0.0,
            "ocr_confidence_score": _clip01(ocr_confidence),
            "laplacian_variance": 0.0,
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
        }

    if weights is None:
        weights = {
            "blur": 0.35,
            "brightness": 0.2,
            "contrast": 0.2,
            "ocr_confidence": 0.25,
        }

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
    lap_var = float(cv.Laplacian(gray, cv.CV_64F).var())
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    blur_score = _normalize_blur_score(lap_var)
    brightness_score = _normalize_brightness_score(brightness_mean)
    contrast_score = _normalize_contrast_score(contrast_std)
    ocr_conf_score = _clip01(ocr_confidence)

    quality = (
        weights["blur"] * blur_score
        + weights["brightness"] * brightness_score
        + weights["contrast"] * contrast_score
        + weights["ocr_confidence"] * ocr_conf_score
    )

    return {
        "quality_score": _clip01(quality),
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "ocr_confidence_score": ocr_conf_score,
        "laplacian_variance": lap_var,
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
    }
