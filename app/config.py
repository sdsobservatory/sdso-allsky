from pydantic_settings import BaseSettings
from app.acquire_fits import AcquireMethod


class Settings(BaseSettings):
    save_images: bool = False
    data_dir: str = "/data"
    interval: int = 3
    camera: str = 'ASI224MC'
    acquire_method: AcquireMethod = AcquireMethod.HTTP
    acquire_base_url: str | None = None
    rotate_180: bool = True
    target_median: int = 1024
    day_to_night_transition_e_per_sec: int = 100_000
    overlay_cardinal_directions: str = 'WSEN'  # Left, Top, Right, Bottom
    overlay_font_size: int = 22
    overlay_title: str | None = None
    overlay_latitude: str | None = None
    overlay_longitude: str | None = None


settings = Settings()
