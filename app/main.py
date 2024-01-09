from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.timelapse import Timelapse
import app.cameras as cameras

try:
    # Get the camera now before fastapi starts so we can exit
    # early if the configuration is incorrect.
    camera = getattr(cameras, settings.camera)()
except:
    raise ValueError(f'Unsupported camera {settings.camera}')


async def configure_timelapse(scheduler: AsyncIOScheduler) -> None:
    global app
    global camera
    timelapse = Timelapse(camera)
    app.state.timelapse = timelapse
    await timelapse.initialize()
    scheduler.add_job(
        timelapse.tick, trigger=IntervalTrigger(seconds=settings.interval), args=(app.state,), id='timelapse'
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler()
    await configure_timelapse(scheduler)
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get('/image')
async def image(request: Request):
    timelapse: Timelapse = request.app.state.timelapse
    return Response(timelapse.jpeg_image_data,
                    media_type='image/jpeg',
                    headers={'content-length': str(len(timelapse.jpeg_image_data))})

