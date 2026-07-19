FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    libxrender1 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user website/backend/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user website/backend/app ./website/backend/app
COPY --chown=user model_artifacts ./model_artifacts

RUN mkdir -p ./website/backend/data

WORKDIR /home/user/app/website/backend

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
