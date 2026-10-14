# SIEM Enrichment Service

Сервис обогащает инциденты из SIEM техниками MITRE ATT&CK. На вход приходит JSON
корреляции, на выходе — тот же JSON плюс подходящие техники и тактики.

## Как это работает

1. Строим поисковый запрос из инцидента (ruleName, title, description, context).
   Русский текст переводим на английский через MarianMT, потому что MITRE
   описан на английском.
2. Ищем похожие техники двумя способами:
   - BM25 по тексту описаний
   - Эмбеддинги (multilingual-e5-small) + косинусное сходство
   Результаты объединяем через RRF.
3. Прогоняем кандидатов через cross-encoder reranker (bge-reranker-v2-m3),
   чтобы отсеять мусор.
4. Добавляем rule-based якорь по `ruleId` (например, `rule-ide-002-*` → DNS-техники).
5. Фильтруем pre-attack тактики (recon, resource development) и применяем
   порог по score.
6. Обогащаем итоговый список названиями техник, тактиками и т.п.

## Стек

- FastAPI + uvicorn
- sentence-transformers (эмбеддинги + reranker)
- transformers + MarianMT (перевод RU→EN)
- bm25s для лексического поиска
- pydantic для моделей
- torch с CUDA (если есть GPU)

## Требования

- Python 3.11
- Docker
- GPU от 6 ГБ VRAM, если хочется скорости. На CPU тоже работает, просто медленнее.

## Запуск через Docker

```bash
docker compose up --build
```
