FROM python:3.10-slim

WORKDIR /app

# system deps (helps TF + plotting libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# copy project
COPY . /app

# install python deps
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# streamlit config
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 7860

# run app
CMD ["streamlit", "run", "streamlit_app/app.py"]