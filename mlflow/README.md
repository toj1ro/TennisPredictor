# Локальный MLflow + MinIO (Чекпоинт 7)

Стек для трекинга экспериментов:

- **MLflow** http://localhost:5000
- **MinIO** http://localhost:9001

Ноутбукам нужен только `MLFLOW_TRACKING_URI=http://localhost:5000` артефакты идут в MinIO через сам сервер.

## Запуск

```bash
cd mlflow
cp .env.example .env
docker compose up -d --build
```

Логин в MinIO (http://localhost:9001) из `.env` (по умолчанию `minioadmin` / `minioadmin`).
