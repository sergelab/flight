# Сборка и запуск

## Требования
- Python 3.12
- Poetry

## Установка
```bash
poetry install
```

## Запуск
```bash
poetry run flight
```

Параметры:
- `--seed <int|random>`
- `--speed <float>`
- `--height-offset <float>`
- `--wireframe`

## Замечания для macOS
- pygame использует системный OpenGL. При проблемах с запуском обновите pygame и убедитесь, что запускаете из GUI-сессии (не из headless окружения).


## OpenGL контекст
Проект требует OpenGL 3.2+ (предпочтительно 3.3 core). На macOS контекст запрашивается через атрибуты pygame.


## Debug overlay
```bash
poetry run flight --debug
```
