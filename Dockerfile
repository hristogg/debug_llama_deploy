# Dockerfile
FROM llamaindex/llama-deploy:main
WORKDIR  /app/code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# This will be passed at build time
COPY src src

RUN ls -l 


EXPOSE 8080
ENV LLAMA_DEPLOY_APISERVER_HOST=0.0.0.0
ENV LLAMA_DEPLOY_API_SERVER_URL="http://127.0.0.1:8080"
ENV LLAMA_DEPLOY_APISERVER_PORT=8080