# Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
   build-essential \
   curl \
   software-properties-common \
   git

COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
   poetry config virtualenvs.create false && \
   poetry install --no-interaction --no-ansi

COPY . .

CMD ["poetry", "run", "streamlit", "run", "web/dashboard/app.py"]