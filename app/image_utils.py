from typing import Tuple
from enum import StrEnum
from io import BytesIO
import numpy as np
from astropy.io import fits
import cv2
from PIL import Image as img
from PIL.Image import Image
from scipy.stats import median_abs_deviation


class ImageFormat(StrEnum):
    JPEG = 'jpeg'
    PNG = 'png'


def to_image(fits_image: fits.ImageHDU) -> Image:
    fits_bayer = fits_image.header['BAYERPAT'].strip()
    cv_bayer = {
        'RGGB': cv2.COLOR_BAYER_RG2BGR,
        'BGGR': cv2.COLOR_BAYER_BG2BGR,
        'GRBG': cv2.COLOR_BAYER_GR2BGR,
        'GBRG': cv2.COLOR_BAYER_GB2BGR,
    }[fits_bayer]
    bgr = cv2.cvtColor(fits_image.data, cv_bayer)
    image = img.fromarray((bgr / 256).astype('uint8'), mode='RGB')
    return image


def to_bytes(image: Image, format: ImageFormat) -> bytes:
    stream = BytesIO()
    image.save(stream, format=format.value or 'jpeg')
    return stream.getvalue()


def green_pixels(image: fits.ImageHDU) -> np.ndarray:
    pattern = image.header['BAYERPAT'].strip()
    data = image.data
    if pattern == 'RGGB':
        green_data_1 = data[0::2, 1::2]
        green_data_2 = data[1::2, 0::2]
        green_data = np.concatenate((green_data_1, green_data_2,)).flatten()
        return green_data
    else:
        raise NotImplementedError


def green_median(image: fits.ImageHDU, bias_median: float = 0) -> float:
    green_data = green_pixels(image) - bias_median
    median = np.median(green_data)
    return median


def green_median_mad(image: fits.ImageHDU, bias_median: float = 0) -> Tuple[float, float]:
    green_data = green_pixels(image) - bias_median
    median = np.median(green_data)
    mad = median_abs_deviation(green_data)
    return median, mad


def mtf(m: float, x: float) -> float:
    if x > 0:
        if x < 1:
            m1 = m - 1.0
            return m1 * x / ((m + m1) * x - m)
        return 1.0
    return 0.0


class STF:

    def __init__(self, c0: float = 0.0, m: float = 0.5, c1: float = 1.0) -> None:
        self.c0 = c0
        self.m = m
        self.c1 = c1

    @staticmethod
    def mtf(m: float, x: float) -> float:
        if x > 0:
            if x < 1:
                m1 = m - 1.0
                return m1 * x / ((m + m1) * x - m)
            return 1.0
        return 0.0

    @staticmethod
    def create(median: float, mad: float) -> 'STF':
        low_clip = -2.8
        target_background = 0.25

        if median < 0.5:
            c0 = max(0.0, min(median + low_clip * mad, 1.0))
            m = STF.mtf(target_background, max(0.0, min(median - c0, 1.0)))
            c1 = 1.0
        else:
            c1 = max(0.0, min(median - low_clip * mad, 1.0))
            m = STF.mtf(max(0.0, min(c1 - median, 1.0)), target_background)
            c0 = 0.0

        return STF(c0, m, c1)


def stretch(image: fits.ImageHDU):
    PIXEL_MAX = 65535

    median = np.median(image.data, axis=None) / PIXEL_MAX
    mad = median_abs_deviation(image.data, axis=None) / PIXEL_MAX
    stf = STF.create(median, mad)

    d = 1.0
    has_clipping = stf.c0 != 0.0 or stf.c1 != 1.0
    has_mtf = stf.m != 0.5
    has_delta = False

    if has_clipping:
        d = stf.c1 - stf.c0
        has_delta = 1 + d != 1

    with np.nditer(image.data, op_flags=['readwrite']) as it:
        for x in it:
            value = x / PIXEL_MAX
            if has_clipping:
                if has_delta:
                    if value <= stf.c0:
                        value = 0
                    elif value >= stf.c1:
                        value = 1.0
                    else:
                        value = (value - stf.c0) / d
                else:
                    value = stf.c0

            if has_mtf:
                value = stf.mtf(stf.m, value)

            x[...] = int(value * PIXEL_MAX)
