
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

WORKDIR /data

ENTRYPOINT ["anatok"]
CMD ["--help"]
