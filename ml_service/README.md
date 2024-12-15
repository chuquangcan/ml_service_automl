# ml_service

run with python 3.10

remember to setup python enviroment

## How to start

ml-api: python ml-api/app/main.py

ml-celery:

```
celery -A ml-celery.app.celery worker --loglevel=info --concurrency=1 -P threads
celery -A ml-celery.app.celery worker --loglevel=info -P solo

```

docker-compose up -d redis rabbitmq

docker-compose build && docker-compose up

view api at docs
