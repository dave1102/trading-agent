# Meme coin scanner and trend paper test (paper only: no wallet, no keys, no real trades).
# Data lives in /app/data, mounted from the server so it survives updates.
# The default command runs the scanner; the trendpaper container overrides it.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-scanner.txt .
RUN pip install --no-cache-dir -r requirements-scanner.txt

COPY pyproject.toml config.toml ./
COPY trading_agent ./trading_agent

# Run as an unprivileged user with a fixed id, so the server folder can be owned by it.
RUN useradd --uid 10001 --create-home scanner && mkdir -p /app/data && chown scanner /app/data
USER scanner

CMD ["python", "-u", "-m", "trading_agent.memescan", "run", "--every", "15", "--track-every", "5", \
     "--log", "data/memecoins/scanner.log"]
