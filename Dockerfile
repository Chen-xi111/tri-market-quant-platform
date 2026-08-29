FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    QUANT_PUBLIC_HOST=localhost \
    QUANT_PUBLIC_MODE=true

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8500

HEALTHCHECK --interval=15s --timeout=3s --start-period=35s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8500/_stcore/health', timeout=2)"

CMD ["python", "scripts/run_all.py", "--host", "0.0.0.0"]
