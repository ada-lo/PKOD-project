import math
import random

import cv2 as cv
import numpy as np


def motion_blur(image, kernel_size=9, angle=0):
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv.warpAffine(kernel, matrix, (kernel_size, kernel_size))
    kernel /= max(kernel.sum(), 1e-6)
    return cv.filter2D(image, -1, kernel)


def gaussian_blur(image, kernel_size=5, sigma=1.2):
    kernel_size = max(3, int(kernel_size) | 1)
    return cv.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def gaussian_noise(image, sigma=12.0):
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def jpeg_compression(image, quality=35):
    ok, encoded = cv.imencode(".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    decoded = cv.imdecode(encoded, cv.IMREAD_COLOR)
    return decoded if decoded is not None else image


def low_light(image, alpha=0.55, beta=-25):
    adjusted = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    hsv = cv.cvtColor(adjusted, cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.85, 0, 255).astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


def perspective_distortion(image, strength=0.08):
    h, w = image.shape[:2]
    dx = int(w * strength)
    dy = int(h * strength)
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([
        [random.randint(0, dx), random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), random.randint(0, dy)],
        [random.randint(0, dx), h - 1 - random.randint(0, dy)],
        [w - 1 - random.randint(0, dx), h - 1 - random.randint(0, dy)],
    ])
    matrix = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(image, matrix, (w, h), borderMode=cv.BORDER_REPLICATE)


def rain_effect(image, streaks=40, length=18, alpha=0.18):
    overlay = image.copy()
    h, w = image.shape[:2]
    for _ in range(streaks):
        x = random.randint(0, max(0, w - 1))
        y = random.randint(0, max(0, h - 1))
        x2 = min(w - 1, x + random.randint(-3, 3))
        y2 = min(h - 1, y + length)
        cv.line(overlay, (x, y), (x2, y2), (210, 210, 210), 1)
    blurred = cv.GaussianBlur(overlay, (3, 3), 0)
    return cv.addWeighted(blurred, alpha, image, 1.0 - alpha, 0)


def fog_effect(image, intensity=0.35):
    h, w = image.shape[:2]
    fog = np.full((h, w, 3), 220, dtype=np.uint8)
    blurred = cv.GaussianBlur(image, (0, 0), sigmaX=10, sigmaY=10)
    return cv.addWeighted(blurred, 1.0 - intensity, fog, intensity, 0)


def downscale(image, scale=0.55):
    h, w = image.shape[:2]
    small = cv.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv.INTER_AREA)
    return cv.resize(small, (w, h), interpolation=cv.INTER_LINEAR)


DEGRADATION_FUNCTIONS = {
    "motion_blur": motion_blur,
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg_compression": jpeg_compression,
    "low_light": low_light,
    "perspective_distortion": perspective_distortion,
    "rain": rain_effect,
    "fog": fog_effect,
    "downscale": downscale,
}


def apply_degradation(image, degradation_name, **kwargs):
    fn = DEGRADATION_FUNCTIONS.get(degradation_name)
    if fn is None or image is None:
        return image
    return fn(image, **kwargs)


def apply_degradation_pipeline(image, degradations):
    output = image.copy()
    for spec in degradations or []:
        if isinstance(spec, str):
            output = apply_degradation(output, spec)
        elif isinstance(spec, dict):
            name = spec.get("name")
            params = spec.get("params", {})
            output = apply_degradation(output, name, **params)
    return output
