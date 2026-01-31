# Сборка и запуск

Этот документ описывает **актуальный** способ запуска приложения и все поддерживаемые CLI-параметры.

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

## Параметры CLI

### Базовые
- `--seed <int|random>` — сид мира (по умолчанию `12345`). `random` выбирает случайный сид.
- `--speed <float>` — **максимальная** скорость полёта (по умолчанию `20.0`).
- `--height-offset <float>` — высота камеры над рельефом (по умолчанию `15.0`).
- `--wireframe` — каркасный режим.
- `--debug` — debug overlay (HUD + логи).

### Геометрия мира и туман
- `--chunk-res <int>` — количество вершин на сторону чанка (по умолчанию `64`).
- `--chunk-size <float>` — размер чанка в мировых единицах (по умолчанию `64.0`).
- `--fog-start <float>` — старт тумана (по умолчанию `150.0`).
- `--fog-end <float>` — конец тумана (по умолчанию `600.0`).

### Производительность и режимы
- `--target-fps <int>` — целевой FPS для адаптивного стриминга чанков (по умолчанию `60`).
- `--auto` — автополёт вперёд (как в ранних версиях, без ручного управления) (по умолчанию **выключено**).
- `--turn-rate <float>` — **legacy** (v0.8): скорость мгновенного поворота, рад/с. В v0.9 используется как fallback для yaw-rate caps при тюнинге (по умолчанию `1.8`).
- `--lod / --no-lod` — LOD-кольца (по умолчанию **выключено**: `--no-lod`).
- `--noise fast|simplex` — режим шума высот (по умолчанию `fast`).

### Управление полётом (v0.9: инерция)
Параметры работают **только** в ручном режиме (когда `--auto` выключен).

**Скорость**
- `--accel <float>` — ускорение к целевой скорости (units/sec^2).
- `--brake <float>` — торможение к целевой скорости (units/sec^2).
- `--drag <float>` — drag скорости (1/sec), выше = быстрее затухает «накат».

**Поворот (yaw)**
- `--yaw-rate-slow <float>` — max yaw rate на низкой скорости (rad/sec).
- `--yaw-rate-fast <float>` — max yaw rate на высокой скорости (rad/sec).
- `--yaw-accel <float>` — угловое ускорение к целевому yaw rate (rad/sec^2).
- `--yaw-decel <float>` — угловое замедление к целевому yaw rate (rad/sec^2).
- `--yaw-drag <float>` — drag yaw rate (1/sec).

**Реакция камеры**
- `--bank-gain <float>` — gain крена в повороте.
- `--bank-max <float>` — max угол крена (radians).
- `--bank-smooth <float>` — сглаживание крена (1/sec).

**Сглаживание цифрового ввода и "вес" камеры**
- `--input-smooth <float>` — сглаживание стрелок (1/sec). Меньше = тяжелее, больше = резче.
- `--cam-yaw-smooth <float>` — лаг yaw камеры (1/sec). Меньше = тяжелее.
- `--climb-rate <float>` — скорость набора/снижения высоты для Q/A (units/sec).
- `--pitch-gain <float>` — визуальный pitch от продольного ускорения.
- `--pitch-max <float>` — max pitch (radians).
- `--pitch-smooth <float>` — сглаживание pitch (1/sec).

### Леса (v0.7)
- `--trees / --no-trees` — включить/выключить леса (по умолчанию **включено**: `--trees`).
- `--tree-density <float>` — множитель плотности деревьев (по умолчанию `1.0`).

## Быстрые сценарии проверки

### Ручное управление (по умолчанию)
```bash
poetry run flight
```
Управление: ↑ вперёд, ↓ назад, ←/→ поворот, Q/A вверх/вниз, Esc выход.

### «Тяжёлый» полёт (больше массы)
```bash
poetry run flight --accel 3.2 --brake 4.6 --drag 0.14 --yaw-accel 1.6 --yaw-decel 2.1 --yaw-drag 1.2 --input-smooth 2.2 --cam-yaw-smooth 2.0
```

### «Резкий» полёт (быстрее реагирует)
```bash
poetry run flight --accel 8 --brake 11 --drag 0.28 --yaw-accel 4.5 --yaw-decel 6 --yaw-drag 2.5 --input-smooth 8 --cam-yaw-smooth 7
```

### Автополёт
```bash
poetry run flight --auto
```

### Debug overlay
```bash
poetry run flight --debug
```

### Отключить леса (сравнение производительности/картинки)
```bash
poetry run flight --no-trees --debug
```

### Включить LOD (экспериментально)
```bash
poetry run flight --lod --debug
```

## Замечания для macOS
- pygame использует системный OpenGL. Запускайте из GUI-сессии (в headless окружениях окно/контекст часто не создаётся).
- Проект рассчитан на OpenGL 3.2+ (предпочтительно 3.3 core). На macOS контекст запрашивается через GL-атрибуты pygame.
