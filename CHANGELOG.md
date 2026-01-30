# CHANGELOG (v0.1.0)

## Hotfix: macOS OpenGL context request
- Добавлен запрос OpenGL 3.3 Core Profile через pygame GL attributes
- Добавлен fallback выбора GLSL версии (330/150) по версии контекста


## Что сделано
- Каркас приложения (pygame + ModernGL), запуск через `poetry run flight`
- Процедурная гористая поверхность на основе OpenSimplex (fBm)
- Бесконечный мир через чанки (коридор 5x8) с подгрузкой по мере полёта
- Фоновая генерация чанков (thread + очереди), дозированная загрузка в GPU
- Камера на "рельсах": движение вперёд, высота следует рельефу со сглаживанием
- Простое освещение (Lambert) + туман по дальности, цвет по высоте
- Документация для PM и технических специалистов + база знаний

## Как проверить
1) `poetry install`
2) `poetry run flight`
3) Наблюдайте непрерывный полёт вперёд без поворотов и без остановки мира.
4) Закройте окно или нажмите `Esc`.

## Изменения в интерфейсе/конфиге
- Добавлен CLI `flight` и параметры `--seed`, `--speed`, `--height-offset`, `--wireframe`

## Hotfix: terrain not visible
- Исправлен порядок индексов (winding) для корректной видимости при backface culling
- Отключён CULL_FACE по умолчанию (можно включить позже после проверки)

## Last fix: запуск + диагностика
- Исправлен debug-fix (World.update_requests не теряется)
- Добавлен debug triangle и периодический лог (z/y/chunks)
- Уточнён запрос OpenGL контекста под macOS

## Variant B (на основе текущей ветки)
- Warmup на старте (предзагрузка чанков)
- Флаг `--debug` включает HUD и логи
- Ускоренная подгрузка чанков первые 2 секунды

## v0.2
- Добавлены контурные линии по высоте (упрощают восприятие рельефа и движения)
- Добавлены CLI параметры: --chunk-res, --chunk-size, --fog-start, --fog-end
- Добавлен документ: docs/tech/apply_archives.md (как применять архивы)
- HUD в --debug показывает параметры мира и тумана

## v0.2.1
- Туман: изменён расчёт на более заметный (smoothstep по forward distance)
- Ускорена генерация чанков: добавлен векторизованный быстрый шум и height_grid

## v0.3.1 (B)
- Двухкольцевой LOD (near/far)
- u_chunk_fade + distance-fade для сглаживания переходов и pop-in
- CLI: --lod/--no-lod, --target-fps, --noise

## v0.3.2 (B hotfix)
- Исправлен fade near/far (near fade-out, far fade-in)
- Добавлен polygon offset при рендере far для уменьшения z-fighting (мерцания)

## v0.3.3 (B hotfix)
- FAR-pass: depth_mask=False + polygon offset (устранение мерцания)
- Шейдер: усилены контуры, добавлен процедурный detail и slope shading
- Blend-band расширен для более мягкого перехода
