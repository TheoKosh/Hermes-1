FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
COPY pyproject.toml ./
COPY hermes_trading ./hermes_trading
COPY state ./state-seed
RUN uv sync
ENV HERMES_TRADING_MODE=paper
# Seed the volume on first boot if empty, then start the worker
CMD ["sh", "-c", "if [ ! -f /app/state/goal.yaml ]; then cp -r /app/state-seed/. /app/state/; fi && uv run python -m hermes_trading.run"]
