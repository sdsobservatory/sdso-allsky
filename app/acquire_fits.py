from enum import Enum
import requests
import asyncio
from io import BytesIO
from astropy.io import fits


class AcquireMethod(Enum):
    HTTP = 1


async def _acquire_fits_http(exposure: float,
                             gain: int,
                             offset: int,
                             base_url: str) -> bytes:
    assert base_url is not None, 'base_url is not defined in settings'
    base_url = base_url.removesuffix('/')
    expose_response = requests.post(f'{base_url}/camera/expose',
                                    json={
                                        'exposure': exposure,
                                        'gain': gain,
                                        'offset': offset,
                                        'is_dark': bool(exposure == 0),
                                    })
    expose_response.raise_for_status()

    # wait for the exposure
    await asyncio.sleep(exposure)

    # poll for complete status
    while True:
        status_response = requests.get(f'{base_url}/camera/status')
        status_response.raise_for_status()
        json = status_response.json()
        if json['status'] == 'complete':
            break
        elif json['status'] == 'error':
            raise ValueError('error exposing camera')

        # TODO: timeout
        await asyncio.sleep(0.2)

    image_response = requests.get(f'{base_url}/camera/image')
    image_response.raise_for_status()
    return image_response.content


async def acquire_fits(method: AcquireMethod,
                       exposure: float,
                       gain: int,
                       offset: int,
                       **kwargs) -> fits.ImageHDU:
    lookup = {
        AcquireMethod.HTTP: _acquire_fits_http
    }

    func = lookup.get(method, None)
    if func is None:
        raise ValueError(f'unsupported acquisition method: {method}')

    fits_bytes = await func(exposure, gain, offset, **kwargs)
    return fits.open(BytesIO(fits_bytes))[0]


if __name__ == '__main__':
    async def main():
        fits_image = await acquire_fits(AcquireMethod.HTTP, .000100, 0, 10, base_url='http://10.10.10.61/')
        fits_image.writeto(f'c:\\tmp\\image-100us.fits', overwrite=True)


    asyncio.run(main())
