# The prototype, packaged so it runs without a local Python 3.14 setup.
#
# Build from the repo root, after building the demo sample:
#   python prototype/scripts/build_demo_sample.py
#   docker build -t momo-fraud-demo .
#   docker run --rm -p 8501:8501 momo-fraud-demo
#
# The 470 MB raw CSV is deliberately not copied. Everything the prototype needs
# is derived: the frozen bundle, one prediction file, and the demo sample.

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by xgboost and lightgbm; the slim image omits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so edits to the app do not invalidate the install layer.
# This is the prototype's own list, not the root requirements.txt: inference
# needs neither JupyterLab nor the training-only learners. See the comments in
# that file for what is left out and why.
COPY prototype/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# The model bundle, the evidence, and the prototype itself.
COPY artifacts/ ./artifacts/
COPY figures/ ./figures/
COPY results/ ./results/
COPY prototype/ ./prototype/

# Only the one prediction file the threshold page reads (about 5 MB). The rest
# of predictions/ is a 210 MB cache the prototype never touches.
COPY predictions/xgboost__unweighted__test__seed42__no_origin_balance.parquet \
     ./predictions/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "prototype/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
