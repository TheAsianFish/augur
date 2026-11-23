# Augur in a container, for running the bot outside GitHub Actions.
#
#   docker build -t augur-forecast:0.1.0 .
#   docker run --rm -v augur-data:/data augur-forecast:0.1.0            # dry run, samples
#   docker run --rm -e METACULUS_TOKEN -e OPENROUTER_API_KEY -e AUGUR_FORECAST_MODELS \
#       -v augur-data:/data augur-forecast:0.1.0 forecast --tournament minibench --submit
#
# Secrets come only from `docker run -e` or `--env-file`; none is baked in.
# The ledger lives on the /data volume (AUGUR_LEDGER=/data/augur.db).

FROM ghcr.io/astral-sh/uv:0.13.9 AS uv

FROM python:3.12-slim AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON=/usr/local/bin/python3.12 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/augur
WORKDIR /src
# Dependencies first, from the lock file, so a code change does not re-resolve them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
# Then the project itself, installed as a wheel (not editable), with its data files.
COPY README.md LICENSE ./
COPY augur ./augur
COPY data/samples ./data/samples
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
LABEL org.opencontainers.image.title="augur-forecast" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.source="https://github.com/TheAsianFish/augur" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 10001 augur \
    && mkdir -p /data \
    && chown augur:augur /data
COPY --from=build /opt/augur /opt/augur
ENV PATH=/opt/augur/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    AUGUR_LEDGER=/data/augur.db
USER augur
WORKDIR /data
VOLUME ["/data"]
ENTRYPOINT ["augur"]
# A dry run: nothing is posted without --submit and METACULUS_TOKEN.
CMD ["forecast", "--tournament", "minibench"]
