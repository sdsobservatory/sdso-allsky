from typing import Tuple, Optional
from starlette.datastructures import State
from app.cameras import Camera
from app.config import settings
from app.acquire_fits import acquire_fits
from app.image_utils import ImageFormat, to_image, to_bytes, green_pixels, green_median, stretch
from app.polynomial_regression import PolynomialRegression
from astropy.io import fits
from threading import RLock
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from pathlib import Path
from datetime import datetime, timedelta
from PIL import ImageDraw
from PIL.Image import Image
import app.metrics as app_metrics


class Timelapse:

    def __init__(self, camera: Camera) -> None:
        self.camera = camera
        self.image_lock = RLock()
        self.image: Optional[Image] = None
        self.jpeg_image_data = bytes()
        self.capture_datetime = datetime.now()
        self.exposure_us = self.camera.exposure_min
        self.gain = self.camera.gain_day
        self.window_size = 30
        self.window_e_per_sec = []
        self.data_dir = Path(settings.data_dir)
        if settings.save_images:
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def _clamp_exposure(self, exposure_us: int) -> int:
        return max(self.camera.exposure_min, min(self.camera.exposure_max, exposure_us))

    def _is_clipped(self, exposure_us: int) -> bool:
        return exposure_us == self.camera.exposure_min

    def _compute_exposure_us_and_gain(self, image: fits.ImageHDU) -> Tuple[int, int]:
        exposure_sec = float(image.header['EXPOSURE'])
        exposure_us = int(exposure_sec * 1e6)
        gain = int(image.header['GAIN'])
        green_data = (green_pixels(image) - self.camera.get_bias_median(exposure_us, gain)) >> self.camera.shift
        median = int(np.median(green_data))
        e_per_sec = median * self.camera.get_e_per_adu(gain) / exposure_sec

        app_metrics.TIMELAPSE_EXPOSURE_METRIC.set(exposure_sec)
        app_metrics.TIMELAPSE_GAIN_METRIC.set(gain)
        app_metrics.TIMELAPSE_E_PER_SEC_METRIC.set(e_per_sec)

        # append to prediction window, remove the oldest entries
        self.window_e_per_sec.append(e_per_sec)
        while len(self.window_e_per_sec) > self.window_size:
            self.window_e_per_sec.pop(0)

        # 3 point minimum for curve fitting
        if len(self.window_e_per_sec) > 3:
            try:
                # predict the next e_per_sec by fitting curves to the previous e_per_sec seen in the window
                if len(self.window_e_per_sec) < 10:
                    next_e_per_sec = self._linear_fit(self.window_e_per_sec)
                elif len(self.window_e_per_sec) < 20:
                    next_e_per_sec = self._linear_ransac_fit(self.window_e_per_sec)
                elif len(self.window_e_per_sec) < 30:
                    next_e_per_sec = self._poly_ransac_fit(self.window_e_per_sec, degree=2)
                else:
                    next_e_per_sec = self._poly_ransac_fit(self.window_e_per_sec, degree=3)
            except:
                print('window_e_per_sec')
                [print(x) for x in self.window_e_per_sec]
                raise
        else:
            next_e_per_sec = e_per_sec

        old_gain = gain
        if e_per_sec > settings.day_to_night_transition_e_per_sec:
            gain = self.camera.gain_day
        else:
            gain = self.camera.gain_night

        # clear the window if the gain was changed
        if old_gain != gain:
            self.window_e_per_sec.clear()

        # zero division safety
        if next_e_per_sec == 0:
            next_e_per_sec = 1

        native_target_median = settings.target_median >> self.camera.shift
        exposure_us = self._clamp_exposure(
            int((native_target_median * self.camera.get_e_per_adu(gain) / next_e_per_sec) * 1e6)
        )

        # upper limit to prevent runaway exposure
        exposure_us = min(exposure_us, int(settings.max_exposure_sec * 1e6))

        app_metrics.TIMELAPSE_EXPOSURE_NEXT_METRIC.set(exposure_us / 1e6)
        app_metrics.TIMELAPSE_GAIN_NEXT_METRIC.set(gain)
        app_metrics.TIMELAPSE_E_PER_SEC_NEXT_METRIC.set(next_e_per_sec)

        return exposure_us, gain

    @staticmethod
    def _linear_fit(data: list[float]) -> float:
        linear = LinearRegression()
        linear.fit(np.arange(len(data)).reshape(-1, 1), data)
        linear_coef = linear.coef_[0]
        linear_inter = linear.intercept_
        next_point = linear_coef * len(data) + linear_inter
        return next_point

    @staticmethod
    def _linear_ransac_fit(data: list[float]) -> float:
        ransac = RANSACRegressor(LinearRegression(), min_samples=10)
        ransac.fit(np.arange(len(data)).reshape(-1, 1), data)
        ransac_coef = ransac.estimator_.coef_[0]
        ransac_inter = ransac.estimator_.intercept_
        next_point = ransac_coef * len(data) + ransac_inter
        return next_point

    @staticmethod
    def _poly_ransac_fit(data: list[float], degree: int = 2) -> float:
        ransac = RANSACRegressor(PolynomialRegression(degree=degree), min_samples=10)
        ransac.fit(np.arange(len(data)).reshape(-1, 1), data)
        polyfn = np.poly1d(ransac.estimator_.coeffs)
        next_point = polyfn(len(data))
        return next_point

    async def acquire(self, exposure_us: int, gain: int) -> fits.ImageHDU:
        self.capture_datetime = datetime.now()
        image = await acquire_fits(settings.acquire_method,
                                   exposure=exposure_us / 1_000_000.0,
                                   gain=gain,
                                   offset=self.camera.offset,
                                   base_url=settings.acquire_base_url)
        return image

    def save_images(self) -> None:
        filename = f'{datetime.now().strftime("%Y-%m-%d %H.%M.%S")}.jpg'
        directory = self.data_dir / 'images' / (datetime.now() - timedelta(hours=12)).strftime("%Y-%m-%d")
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / filename

        with self.image_lock:
            with open(image_path, 'wb') as f:
                f.write(self.jpeg_image_data)

    def paint_overlay(self) -> None:
        if self.image is None:
            return

        with self.image_lock:
            text_args = {
                'fill': 'white',
                'font_size': settings.overlay_font_size,
                'align': 'center',
                'stroke_fill': '#333333',
                'stroke_width': 2,
            }
            w = self.image.width
            h = self.image.height
            d = ImageDraw.Draw(self.image)

            text_spacing = settings.overlay_font_size // 4
            bottom_left_anchor = h - text_spacing
            bottom_right_anchor = h - text_spacing

            if settings.overlay_title:
                d.text((w // 2, text_spacing), settings.overlay_title, anchor='mt', **text_args)

            # bottom left overlay items, drawn from bottom -> top

            d.text((10, bottom_left_anchor), datetime.now().strftime("%Y-%m-%d %H-%M-%S"), anchor='lb', **text_args)
            bottom_left_anchor -= settings.overlay_font_size + text_spacing

            if settings.overlay_longitude:
                d.text((10, bottom_left_anchor), settings.overlay_longitude, anchor='lb', **text_args)
                bottom_left_anchor -= settings.overlay_font_size + text_spacing

            if settings.overlay_latitude:
                d.text((10, bottom_left_anchor), settings.overlay_latitude, anchor='lb', **text_args)
                bottom_left_anchor -= settings.overlay_font_size + text_spacing

            # bottom right overlay items, drawn from bottom -> top

            if self.exposure_us < 1000:
                exposure = f'{self.exposure_us:.0f} us'
            elif self.exposure_us < 1e6:
                exposure = f'{self.exposure_us / 1e3:.0f} ms'
            else:
                exposure = f'{self.exposure_us / 1e6:.3f} s'

            d.text((w - text_spacing, bottom_right_anchor), f'Gain {self.gain:>4}', anchor='rb', **text_args)
            bottom_right_anchor -= settings.overlay_font_size + text_spacing

            d.text((w - text_spacing, bottom_right_anchor), exposure, anchor='rb', **text_args)
            bottom_right_anchor -= settings.overlay_font_size + text_spacing

    async def initialize(self) -> None:
        print('initializing auto exposure')
        min_median = 0x80 >> self.camera.shift
        max_median = 0xC000 >> self.camera.shift
        exposure_us = 500_000
        gain = self.camera.gain_night
        median = 0
        native_target_median = settings.target_median >> self.camera.shift

        # Determine a starting exposure
        while True:
            exposure_us = self._clamp_exposure(exposure_us)
            if self._is_clipped(exposure_us):
                if gain == self.camera.gain_night:
                    gain = self.camera.gain_day
                    continue
                break
            print(f'exposing for {exposure_us}us at {gain} gain')
            image = await self.acquire(exposure_us, gain)
            median = green_median(image, self.camera.get_bias_median(exposure_us, gain))
            median = int(median) >> self.camera.shift
            print(f'{self.camera.adc_bit_depth}-bit median = {median}')
            if min_median < median < max_median:
                break
            elif median < min_median:
                exposure_us *= 4
            elif median > max_median:
                exposure_us //= 4

        # calculated exposure to hit the target median
        native_bias_median = self.camera.get_bias_median(exposure_us, gain) >> self.camera.shift
        exposure_us = self._clamp_exposure(
            int(exposure_us * (native_target_median + native_bias_median) / max(1, median)))

        # One iteration of refinement
        image = await self.acquire(exposure_us, gain)
        optimal_exposure_us, optimal_gain = self._compute_exposure_us_and_gain(image)
        self.exposure_us = optimal_exposure_us
        self.gain = optimal_gain
        self.window_e_per_sec = []

        print('initialization complete')

    @staticmethod
    async def tick(state: State) -> None:
        timelapse: 'Timelapse' = state.timelapse

        print(f'exposure_us = {timelapse.exposure_us}, gain = {timelapse.gain}')

        image = await timelapse.acquire(exposure_us=timelapse.exposure_us, gain=timelapse.gain)
        median = green_median(image, timelapse.camera.get_bias_median(timelapse.exposure_us, timelapse.gain))
        print(f'16-bit median without bias  = {median}')
        if timelapse.window_e_per_sec:
            print(f'e-per-sec = {timelapse.window_e_per_sec[-1]}')

        optimal_exposure_us, optimal_gain = timelapse._compute_exposure_us_and_gain(image)
        timelapse.exposure_us = optimal_exposure_us
        timelapse.gain = optimal_gain

        stretch(image)

        output_image = to_image(image)
        if settings.rotate_180:
            output_image = output_image.rotate(180)

        with timelapse.image_lock:
            timelapse.image = output_image
            timelapse.paint_overlay()
            timelapse.jpeg_image_data = to_bytes(output_image, ImageFormat.JPEG)
            if settings.save_images:
                timelapse.save_images()
