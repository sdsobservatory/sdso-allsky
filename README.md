# SDSO Allsky

A timelapse webapp.

Camera control via [SDSO Skycam](https://github.com/sdsobservatory/skycam). An image is taken ~60s
with automatic gain control using a custom algorithm to maintain brightness.

## Building Container

```shell
docker build -t registry.local.sdso.space/sdso-allsky:latest .
docker push registry.local.sdso.space/sdso-allsky:latest
```
