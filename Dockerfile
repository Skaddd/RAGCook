FROM python:3.11-slim as builder


ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

RUN apt-get update && apt-get install -y curl git cmake build-essential \
    && pip install --no-cache-dir --upgrade pip

# Should not be pip but directly CURL from poetry
RUN pip install --no-cache-dir poetry==2.0.1

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root && rm -rf $POETRY_CACHE_DIR

RUN git clone https://github.com/ggerganov/llama.cpp



WORKDIR /app/llama.cpp

# Create the build directory and compile the project
RUN cmake -B build \
    && cmake --build build --config Release

# Might not be working
RUN pip install llama-cpp-python
# The runtime image, used to just run the code provided its virtual environment
FROM python:3.11-slim-buster as runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

COPY src ./src
COPY conf ./conf
COPY data ./data
COPY weights ./weights

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

