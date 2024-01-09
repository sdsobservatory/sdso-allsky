FROM python:3.11-slim
LABEL authors="Alex Helms"

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]