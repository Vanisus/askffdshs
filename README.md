# Few-shot инспекция деталей через HTTP

FastAPI обслуживает каталог эталонов, построение профилей, калибровку и поиск
аномалий. Один живой Inspector и одна замороженная нейросеть на процесс;
отдельного обучения под каждый part_id нет.

## Запуск

Python 3.11/3.12. Установите совместимые torch/torchvision для вашего CUDA, затем:

```bash
pip install -r requirements.txt
export INSPECTION_DEVICE=cuda
export INSPECTION_DATA_DIR=./data
# Задайте собственный длинный случайный ключ перед публикацией сервиса.
export INSPECTION_API_KEY=your-secret-key
python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1
```

Для машины без GPU: INSPECTION_DEVICE=cpu. PowerShell: $env:INSPECTION_DEVICE="cpu".
Без ключа авторизация отключена: такой режим только для изолированного localhost.
Health и Swagger/OpenAPI доступны без ключа; все операции с деталями и артефактами
защищены, если ключ задан. Swagger: http://localhost:8000/docs; нажмите Authorize.
Первый build/inspect может скачивать pretrained-веса. model_loaded в health
не гарантирует наличие CUDA/весов до их реальной загрузки.

## Полный HTTP-цикл

Загрузки multipart/form-data: image — фото, необязательная mask — бинарная PNG
с фоном 0 и деталью >0. Размер маски совпадает с EXIF-ориентированным фото.
Без маски используется сегментация однотонного стола.

```bash
BASE=http://localhost:8000
# Повторить для каждого эталона. part_id создаётся автоматически.
curl --fail-with-body -H "X-API-Key: $INSPECTION_API_KEY" \
  -F image=@./photos/ref1.png "$BASE/parts/detail_001/references"
# При необходимости добавить: -F mask=@./masks/ref1.png
curl --fail-with-body -X POST -H "X-API-Key: $INSPECTION_API_KEY" \
  "$BASE/parts/detail_001/build"
# Отдельные годные кадры, НЕ использованные как эталоны:
curl --fail-with-body -H "X-API-Key: $INSPECTION_API_KEY" \
  -F images=@./validation/normal1.png -F images=@./validation/normal2.png \
  "$BASE/parts/detail_001/calibrate"
curl --fail-with-body -H "X-API-Key: $INSPECTION_API_KEY" \
  -F image=@./photos/current.png "$BASE/parts/detail_001/inspect"
```

Ответ inspect: inspection_id, decision, defects (bbox_xyxy, контуры, score),
качество alignment и относительные URL artifacts. Скачивать BASE + URL,
с тем же X-API-Key. Сохраняются result.json, overlay.png, heatmap.npy float32
и inspection_mask.png. Bbox: [x0,y0,x1,y1], правый/нижний край исключены;
координаты исходного EXIF-ориентированного фото, не crop.

## Маршруты

| Метод | Путь | Назначение |
|---|---|---|
| GET | /health | Состояние, model_loaded, busy, authentication_enabled |
| GET | /parts?limit=20&offset=0 | Детали и количество эталонов |
| GET | /parts/{part_id}/references | ID эталонов, limit/offset |
| POST | /parts/{part_id}/references | Добавить image, опционально mask |
| POST | /parts/{part_id}/build | Построить/перестроить профиль |
| GET | /parts/{part_id}/profile | Проверенный активный профиль |
| POST | /parts/{part_id}/calibrate | images: отдельные годные фото |
| POST | /parts/{part_id}/inspect | Проверить image, опционально mask |
| GET | /inspections/{inspection_id} | Сохранённый результат с URL |
| GET | /inspections/{inspection_id}/{artifact} | Скачать разрешённый артефакт |

part_id: 1–80 ASCII букв/цифр/подчёркиваний/дефисов, первый символ — буква/цифра.
Коды: 201 эталон, 200 успех, 401 auth, 404 неизвестная деталь/результат,
409 отсутствующий/устаревший профиль, 413 размер, 422 фото/маска/alignment,
503 worker занят (Retry-After: 1), 500 внутренняя ошибка (подробности в logs).

## Конкурентность и безопасность

Build/calibrate/inspect — синхронные HTTP-операции, не фоновые задания.
Тяжёлые операции в thread pool; event loop и health остаются отзывчивыми.
Один тяжёлый запрос на процесс, следующие получают 503 вместо неограниченной
очереди и OOM. Используйте один worker на GPU: каждый процесс имеет свою модель.
FileLock защищает запись каталога; межпроцессного GPU-планировщика нет.
Задайте таймауты reverse proxy с учётом первого скачивания и построения профиля.

Лимиты файлов и всего multipart, включая chunked, проверяются сервером.
Имена загруженных файлов не используются как пути. Неудачная инспекция очищает
частичные артефакты. Подробности внутренних ошибок не выдаются клиенту.
API-key общий, не multi-tenant auth. Для публикации нужны HTTPS/reverse proxy,
rate limiting, timeouts, мониторинг диска и политика retention. Удаление результатов,
очередь заданий и строгие Pydantic response-модели в эту версию не входят.

## Настройки окружения

| Переменная | По умолчанию |
|---|---|
| INSPECTION_DEVICE | cuda |
| INSPECTION_DATA_DIR | ./data |
| INSPECTION_API_KEY | не задан |
| INSPECTION_BACKBONE | dinov2_vits14 |
| INSPECTION_TILE_SIZE | 448 |
| INSPECTION_TILE_OVERLAP | 112 |
| INSPECTION_LOCAL_RADIUS | 1 |
| INSPECTION_THRESHOLD | 0.15 |
| INSPECTION_REFERENCE_QUANTILE | 0.25 |
| INSPECTION_MIN_PART_AREA | 5000 |
| INSPECTION_MIN_DEFECT_AREA | 30 |
| INSPECTION_EDGE_MARGIN | 4 |
| INSPECTION_MAX_UPLOAD_BYTES | 52428800, 50 MiB на файл |
| INSPECTION_MAX_REQUEST_BYTES | 134217728, 128 MiB на multipart |
| INSPECTION_MAX_CALIBRATION_IMAGES | 20 |
| DINO_HUB_DIR | локальный проверенный DINOv2 checkout, если задан |

Конфигурация HTTP передаётся в тот же Inspector, что CLI. Несовместимый профиль
требует rebuild. Старые prepare-reference, align, add-reference, build-profile,
calibrate, inspect сохранены: python -m app.main --help.

## Docker CPU

```bash
docker build -t part-inspection .
docker volume create inspection-data
docker volume create inspection-models
docker run --rm -p 127.0.0.1:8000:8000 \
  -e INSPECTION_API_KEY="$INSPECTION_API_KEY" \
  -v inspection-data:/data -v inspection-models:/models part-inspection
```

CPU-only образ, непривилегированный пользователь UID 10001, healthcheck.
Для GPU нужны CUDA PyTorch, NVIDIA Container Toolkit и доступ к GPU;
одного INSPECTION_DEVICE=cuda недостаточно. Веса кэшируются в /models.
Bind-mounted каталоги должны разрешать запись UID 10001.
Docker daemon в проверочной среде отсутствовал: сборка образа не проверена.

## Ограничения контроля

Score — расстояние признаков, не вероятность. Это поиск аномалий, не классификатор
типов царапин/пыли. 4К обрабатывается тайлами; микродефекты 1–2 px не гарантированы.
Нужны контролируемые свет/ракурс и годные эталоны. Кромки/сколы вне пересечения
масок требуют отдельного геометрического контроля. no_candidates не означает
сертифицированную годность: production_acceptance=not_validated.
Один эталон без калибровки даёт inconclusive. Калибровка нормальными кадрами
не измеряет recall на реальном браке. Industrial accuracy и CUDA latency не измерены.

## Проверки

```bash
pip install -r requirements-dev.txt
python -m ruff check app tests prototype.py
python -m compileall -q app tests prototype.py
python -m pytest -q
RUN_PRETRAINED_TESTS=1 TEST_BACKBONE=resnet18 python -m pytest -q -m integration
```

HTTP-тесты: полный цикл с маской, калибровка, защита артефактов, ошибочные/устаревшие
профили, лимиты файлов/chunked-тела, конкурентный 503, отзывчивость health,
очистка частичных результатов. Искусственные сцены/encoder test double только
для функциональной проверки, не промышленного benchmark.
