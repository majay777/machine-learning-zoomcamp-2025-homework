FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/



ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./

RUN uv sync --locked

COPY "question-4.py"  "pipelilne_v1.bin"./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "question-4:app", "--host", "0.0.0.0", "--port", "9696"]