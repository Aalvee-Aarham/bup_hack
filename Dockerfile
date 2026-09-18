FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000 WEB_CONCURRENCY=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py llm.py solver.py ./
COPY static ./static

RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=2)"

# One worker on purpose: LLM quota is the bottleneck, and one process shares rate budgets,
# the note cache and request coalescing. Raising WEB_CONCURRENCY splits budgets per worker.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port $PORT --workers $WEB_CONCURRENCY --no-access-log --timeout-keep-alive 30"]
