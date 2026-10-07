# Навигация по проекту

Этот документ отвечает на практический вопрос: где искать постановку задачи,
рабочий код, сценарии экспериментов, данные и результаты. Актуально на
2026-10-07 после G14-C-R: однофакторный видеосрез валиден, но один шумный
кадр с ошибкой 0,2067 м приняли два из пяти seed. Остальные четыре фактора
дали ноль ложных принятий.

Термины проекта расшифрованы в [тезаурусе](glossary.md). В частности, `Teach`
означает первичную съёмку с известной разметкой, а `Repeat` — повторный кадр,
который алгоритм должен привязать к эталону.

## С чего начать

Для знакомства с проектом достаточно пройти четырнадцать документов в таком порядке:

1. [README](../README.md) — задача, текущие результаты и быстрый запуск.
2. [Дорожная карта](project-roadmap.md) — принятые решения, ограничения и
   ближайшие гейты.
3. [Отчёт G14-C-R](synthetic-3d-video-quality-report.md) — разрешение, шум,
   JPEG, размытие, псевдо-OSD и пограничное ложное принятие шумного кадра.
4. [Отчёт G14-A2-R](synthetic-3d-calibration-uncertainty-repair-report.md) —
   валидная парная проверка калибровочной ошибки и отрицательный гейт безопасности.
5. [Отчёт G14-A](synthetic-3d-lens-distortion-report.md) — ложные принятия при
   пропущенной дисторсии и эффект известной калибровки.
6. [Отчёт G13-R](synthetic-3d-texture-ambiguity-report.md) — наблюдаемый отказ на
   неоднозначной Teach-текстуре и его цена доступностью.
7. [Отчёт G13](synthetic-3d-texture-class-report.md) — безопасный отказ на
   гладкой земле и опасное ложное принятие на повторяющихся рядах.
8. [Отчёт G12](synthetic-3d-pose-interactions-report.md) — взаимодействие
   пограничной позы, качества кадра и ошибок Teach-маски на новых seed.
9. [Отчёт G11](synthetic-3d-pose-sweep-report.md) — границы бокового
   базиса, высоты и наклона повторной камеры.
10. [Отчёт G10](synthetic-3d-clutter-report.md) — устойчивость к статичным
   объектам над плоской землёй.
11. [Отчёт G9-R](synthetic-3d-relative-displacement-report.md) — актуальная
   продуктовая постановка и точность вектора «цель → попадание».
12. [Отчёт G9](synthetic-3d-terrain-pose-report.md) — граница плоской
   метрической модели на гладком рельефе и отрицательный итог G9.
13. [Отчёт G8](synthetic-3d-annotation-robustness-report.md) — граница
   устойчивости Teach-разметки и отрицательный итог G8.
14. [План 3D-гейта](synthetic-3d-gate-plan.md) — исходная история G0–G13 и
   общая логика синтетического стенда.

Если нужно запустить проект, а не разбираться в истории экспериментов, см.
раздел [«Частые команды»](#частые-команды) ниже.

Для продолжения ветки многовидовой проверки Teach-разметки сначала прочитать
[handoff 3D-реконструкции](3d-reconstruction-refrigerator.md): там сохранены
ресурсный аудит, решение «в холодильнике» и точка входа нового чата.

## Карта верхнего уровня

```text
aerial-map-measurement/
├── apps/                 интерактивные приложения
├── data/                 описания и локальное размещение наборов данных
├── docs/                 планы, отчёты и учебные пояснения
├── experiments/
│   ├── configs/          зафиксированные параметры экспериментов
│   └── feasibility/      воспроизводимые исследовательские запуски
├── outputs/              локальные результаты и кэши запусков
├── scripts/              установка инструментов, загрузка данных, Blender
├── src/aerial_mapper/    повторно используемая логика проекта
├── tests/                автоматические тесты пакета
├── AGENTS.md             правила работы над учебным проектом
├── README.md             входная страница проекта
├── LICENSE               лицензия собственного кода и документации
├── THIRD_PARTY.md        происхождение и лицензии внешних компонентов
├── pyproject.toml        зависимости и настройки Python-инструментов
└── uv.lock               точные версии воспроизводимого окружения
```

Локальные каталоги `.venv/` и `.tools/` создаются при установке окружения.
Первый содержит Python-зависимости, второй — portable-версию Blender. Оба
игнорируются Git и не являются исходным кодом проекта.

## Корневые файлы

| Путь | Назначение |
| --- | --- |
| [README.md](../README.md) | Краткая постановка, состояние, ограничения, запуск и ссылки на основные документы. |
| [AGENTS.md](../AGENTS.md) | Обязательные правила для участников и AI-ассистентов: учебная цель, критическая проверка идей, стиль кода и воспроизводимость. |
| [pyproject.toml](../pyproject.toml) | Диапазон Python, прямые зависимости, группы разработки, настройки pytest и Ruff. |
| [uv.lock](../uv.lock) | Зафиксированное дерево зависимостей; изменяется вместе с `pyproject.toml`. |
| [LICENSE](../LICENSE) | Лицензия `GPL-3.0-or-later` для собственного содержимого репозитория. |
| [THIRD_PARTY.md](../THIRD_PARTY.md) | Отдельные условия Blender, библиотек, моделей и наборов данных. |
| `.gitignore` | Правила, исключающие окружение, тяжёлые данные, модели и результаты экспериментов. |

## Рабочая логика: `src/aerial_mapper/`

Код в этом каталоге является библиотечной частью проекта. Если функция нужна
более чем одному эксперименту, её следует переносить сюда, а не копировать
между файлами в `experiments/`.

### Привязка и контроль качества

| Модуль | Что делает |
| --- | --- |
| `alignment.py` | Ищет локальные признаки SIFT, сопоставляет их и оценивает гомографию методом RANSAC. При недостатке доказательств возвращает явный отказ. |
| `quality.py` | Считает наблюдаемые без знания правильного ответа показатели: покрытие кадра, распределение inlier-точек и устойчивость оценки. |
| `evaluation.py` | Сравнивает найденную гомографию со скрытой синтетической истиной на независимых контрольных точках. |
| `visualization.py` | Строит наложения, диагностические изображения и визуализации геометрии. |

Гомография — матрица 3 × 3, связывающая две проекции одной плоскости.
RANSAC — устойчивый метод оценки, который допускает часть ошибочных пар.
Inlier-точки — пары, согласующиеся с найденной моделью. Подробнее:
[тезаурус](glossary.md).

### Перевод в метры

| Модуль | Что делает |
| --- | --- |
| `measurement.py` | Переводит пиксели через геопривязанный эталон в метрические координаты и измеряет отрезки и полигоны. |
| `measurement_evaluation.py` | Сравнивает измерения со скрытой истиной на контрольных фигурах. |
| `metric_recovery.py` | Собирает безопасную цепочку «Repeat RGB → эталон → локальные метры» и запрещает измерение неподтверждённых поверхностей. |
| `perspective_metric.py` | Калибрует перспективный Teach-кадр по известным соответствиям «пиксель ↔ координата земли» и переносит метрику на Repeat. |
| `reference_grid.py` | Описывает регулярный геопривязанный растр и преобразования его пикселей в проекционные координаты. |

### Контрольная истина и синтетика

| Модуль | Что делает |
| --- | --- |
| `synthetic.py` | Создаёт плоские кадры с заранее известной гомографией. |
| `synthetic_3d.py` | Проверяет описание Blender-сцены, запускает Blender отдельным процессом и анализирует рендеры. Сам `bpy` сюда не импортируется. |
| `synthetic_robustness.py` | Детерминированно ухудшает Repeat-кадр: экспозиция, тень, размытие, разрешение и другие факторы G4–G6. |
| `teach_annotation.py` | Строит контролируемые ошибки границы, полноты и чистоты Teach-маски и считает их относительно скрытой истины. |
| `surface_evaluation.py` | Раздельно оценивает одну гомографию на земле и крышах, чтобы видеть ошибку от параллакса. |
| `dense_ground_truth.py` | Проверяет гомографию по плотной трёхмерной контрольной истине OrthoLoC. |
| `terrain_geometry.py` | Задаёт воспроизводимые формы рельефа и строит их метрическую полигональную сетку. |
| `terrain_evaluation.py` | Независимо оценивает ошибку точек и длин на рельефе, не переиспользуя точки подгонки. |
| `relative_measurement.py` | Считает вектор «цель → попадание» и раздельно оценивает ошибку компонент, длины, направления и знака. |
| `clutter_geometry.py` | Детерминированно размещает камни, пни и кусты в метрах без зависимости от Blender. |
| `ground_texture.py` | Детерминированно создаёт богатую, низкодетальную и периодическую текстуры земли без зависимости от Blender. |
| `texture_ambiguity.py` | Измеряет долю почти одинаковых SIFT-признаков в удалённых частях Teach и выдаёт предварительный отказ. |
| `camera_distortion.py` | Искажает и исправляет RGB и точки по модели Brown--Conrady с проверяемым циклом координат. |
| `radiance_camera.py` | Переводит настройки Blender в pinhole intrinsics, проверяет camera-to-world и строит мировые пиксельные лучи. |
| `interaction_evaluation.py` | Присваивает продуктовый исход с учётом решения gate, ошибки вектора и ошибок знака. |

### Поиск места на CLOUD

| Модуль | Что делает |
| --- | --- |
| `cloud_dataset.py` | Читает кадры и телеметрию CLOUD. Координаты используются экспериментом для контроля, но не подаются визуальному алгоритму. |
| `place_retrieval.py` | Строит глобальное описание кадра с помощью DINOv2-S и VLAD и ранжирует похожие Teach-кадры. |
| `place_verification.py` | Проверяет короткий список кандидатов геометрией и либо выбирает один Teach-кадр, либо отказывается. |

`src/aerial_mapper/__init__.py` задаёт описание пакета и его версию.

## Приложение: `apps/`

`apps/poc_web.py` — учебный веб-интерфейс на Gradio для плоского прототипа.
Он показывает преобразование, контрольные точки и метрики. Это средство
демонстрации и ручной проверки, а не отдельная реализация алгоритма.

Запуск и устройство интерфейса описаны в
[документе Gradio PoC](gradio-poc.md).

## Эксперименты: `experiments/`

### `experiments/configs/`

JSON-файлы фиксируют входы, случайные seed, параметры сцен и пороги до запуска:

| Конфигурация | Ступень |
| --- | --- |
| `synthetic_3d_g0_smoke.json` | G0: связность Blender-пайплайна и систем координат. |
| `synthetic_3d_g1_metric.json` | G1: восстановление расстояний на идеальной плоскости. |
| `synthetic_3d_g2_height_sweep.json` | G2: влияние высоты объектов и параллакса. |
| `synthetic_3d_g3_g6_robustness.json` | G3–G6: видимая земля, искажения, комбинации и отложенные сцены. |
| `synthetic_3d_g7_teach_repeat.json` | G7: перспективный размеченный Teach и неизвестный Repeat RGB. |
| `synthetic_3d_g8_annotation_robustness.json` | G8: замороженные ошибки Teach-маски, шум метрических кликов и критерии. |
| `synthetic_3d_g9_terrain_pose.json` | G9: замороженный pilot рельефа и позы камеры. |
| `synthetic_3d_g9_boundary_followup.json` | G9: независимое уточнение границы широкого холма. |
| `synthetic_3d_g9r_relative_displacement.json` | G9-R: векторы промаха, картографические реперы и диагностические полосы ошибки. |
| `synthetic_3d_g10_clutter.json` | G10: pilot числа мелких 3D-объектов. |
| `synthetic_3d_g10r_relative_clutter.json` | G10: относительная оценка pilot. |
| `synthetic_3d_g10_clutter_boundary.json` | G10: follow-up крупных объектов. |
| `synthetic_3d_g10r_relative_clutter_boundary.json` | G10: относительная оценка follow-up. |
| `synthetic_3d_g11_pose_*.json` | G11: основные и post-hoc sweep бокового базиса, высоты и наклона. |
| `synthetic_3d_g11r_relative_*.json` | G11: завершённые оценки вектора «цель → попадание». |
| `synthetic_3d_g12_interaction_scenes.json` | G12: три новые сцены с независимыми seed текстуры и объектов. |
| `synthetic_3d_g12_pose_interactions.json` | G12: замороженные позы, ухудшения кадра, ошибка маски и критерии. |
| `synthetic_3d_g13_texture_classes.json` | G13: согласованные seed, классы текстуры, объекты и позы. |
| `synthetic_3d_g13_texture_evaluation.json` | G13: продуктовый вектор и заранее заданные критерии по классам. |
| `synthetic_3d_g13r_ambiguity.json` | G13-R: frozen-пороги самопохожести и девять новых holdout-сцен. |
| `synthetic_3d_g14a_lens_distortion.json` | G14-A: уровни `k1`, точная коррекция и заранее заданные критерии. |
| `synthetic_3d_g14a2_calibration_uncertainty.json` | G14-A2: слабые `k1` и ошибка оценённой калибровки. |
| `synthetic_3d_g14a2r_paired_stability.json` | G14-A2-R: пять парных seed и раздельные критерии валидности и безопасности. |
| `synthetic_3d_g14b_rolling_shutter.json` | G14-B: строковые позы, движения и frozen-критерии repair. |
| `synthetic_3d_g14c_video_quality.json` | G14-C: однофакторные уровни качества видеотракта и парная stability-проверка. |
| `nerf_3dgs_prelesson_export.json` | Подготовительный экспорт четырёх камер, train/test split и строгие пороги проекции. |

Конфигурации — часть протокола. Менять их после просмотра итоговой выборки
нельзя без новой версии эксперимента и явного объяснения.

### `experiments/feasibility/`

Здесь находятся запускаемые исследовательские сценарии. Они собирают библиотечные
функции, читают данные, сохраняют таблицы и строят отчёты.

Плоская синтетика и метрика:

- `clean_alignment_grid.py` — поворот и искусственная перспектива;
- `scale_sweep.py` — устойчивость к изменению масштаба;
- `spatial_alignment_grid.py` — разные положения кадра на эталоне;
- `metric_measurement_poc.py` — контроль координат, длин и площадей.

Реальные данные и поиск места:

- `ortholoc_real_data_spike.py` — два OrthoLoC-примера с плотной истиной;
- `cloud_teach_repeat_smoke_v2.py` — известные пары двух проходов;
- `cloud_dinov2_vlad_retrieval_smoke.py` — поиск кадра по базе;
- `cloud_sequence_retrieval_smoke.py` — проверка простого временного фильтра;
- `cloud_end_to_end_gate.py` — зафиксированный сквозной CLOUD-гейт.

Blender и ступени G0–G14-C:

- `synthetic_3d_smoke.py` — G0;
- `synthetic_3d_metric_recovery.py` — G1;
- `synthetic_3d_height_sweep.py` — G2;
- `synthetic_3d_robustness_suite.py` — G3–G6;
- `synthetic_3d_teach_repeat_gate.py` — G7;
- `synthetic_3d_annotation_robustness.py` — G8;
- `synthetic_3d_terrain_pose.py` — G9, G10 и pose-sweep G11;
- `synthetic_3d_relative_displacement.py` — G9-R и относительная оценка G10–G11;
- `synthetic_3d_pose_interactions.py` — G12--G13: совместная оценка позы, качества, маски и класса текстуры.
- `synthetic_3d_texture_ambiguity.py` — G13-R: Teach-самопохожесть, holdout и переоценка исходов G13.
- `synthetic_3d_lens_distortion.py` — G14-A: дисторсия Repeat, точная коррекция и метрическая оценка.
- `synthetic_3d_calibration_uncertainty.py` — G14-A2: малые уровни и ошибка `k1`.
- `synthetic_3d_calibration_uncertainty_repair.py` — G14-A2-R: парная многосидовая проверка stability.
- `synthetic_3d_rolling_shutter.py` — G14-B-R: строковые позы движущейся камеры.
- `synthetic_3d_video_quality.py` — G14-C-R: разрешение, шум, JPEG, размытие и псевдо-OSD.
- `export_radiance_field_dataset.py` — экспорт камер и масок в Nerfstudio/Synthetic NeRF с проверкой проекций.

## Инструментальные скрипты: `scripts/`

| Файл | Назначение |
| --- | --- |
| `blender_generate_terrain_scene.py` | Создаёт G9 mesh-рельеф и плотную ray-cast истину видимости. |
| `install_blender.sh` | Скачивает и проверяет зафиксированную portable-версию Blender без системной установки. |
| `blender.sh` | Единая точка запуска локального Blender, включая пакетный режим. |
| `blender_generate_scene.py` | Выполняется встроенным Python Blender и создаёт сцену, камеры, рендеры и контрольные слои. |
| `open_blender_scene.sh` | Открывает сгенерированный `scene.blend` в графическом интерфейсе. |
| `download_reference.py` | Воспроизводит временный геопривязанный эталон для плоского PoC. |
| `download_ortholoc_demo.py` | Загружает разрешённые демонстрационные файлы OrthoLoC и проверяет контрольные суммы. |
| `download_cloud_trial.py` | Загружает выбранный CLOUD trial по его манифесту. |

Для первого знакомства с интерфейсом Blender используйте
[быстрый старт](blender-quickstart.md).

## Данные: `data/`

[Инструкция по данным](../data/README.md) описывает происхождение и локальную
структуру наборов.

- `data/manifests/` — небольшие версионируемые JSON-описания источников,
  лицензий, URL, контрольных сумм и ожидаемых файлов.
- `data/reference/` — локальные эталонные растры.
- `data/queries/` — локальные Repeat-кадры и реальные демонстрационные данные.
- `data/models/` — локальный кэш весов моделей.

Содержимое последних трёх каталогов, кроме файлов `.gitkeep`, не коммитится.
Это защищает репозиторий от тяжёлых файлов, неясных лицензий и чувствительных
координат. Чтобы другой человек воспроизвёл опыт, нужно обновить манифест и
скрипт загрузки, а не добавлять локальный набор напрямую.

## Результаты: `outputs/`

`outputs/` содержит сгенерированные CSV, JSON, PNG, `.blend` и кэши
признаков. Каталог игнорируется Git, кроме `.gitkeep`.

Основные семейства путей:

- `outputs/synthetic_3d/g0_smoke/` — G0;
- `outputs/synthetic_3d/g1_metric/` — G1;
- `outputs/synthetic_3d/g2_height_sweep/` — G2;
- `outputs/synthetic_3d/g3_g6_robustness/` — G3–G6;
- `outputs/synthetic_3d/g7_teach_repeat/` — G7;
- `outputs/synthetic_3d/g8_annotation_robustness/` — G8;
- `outputs/cache/` — повторно используемые признаки и промежуточные данные;
- файлы `cloud_*`, `ortholoc_*`, `scale_*` и `spatial_*` — результаты
  соответствующих сценариев из `experiments/feasibility/`.

Результат в `outputs/` не является источником истины сам по себе. Для
воспроизводимости рядом должны существовать код запуска, конфигурация и
документ с интерпретацией. Повторный запуск может перезаписать локальные
артефакты.

## Документация: `docs/`

Документы сгруппированы по назначению.

Ориентация и методика:

- [project-navigation.md](project-navigation.md) — этот документ;
- [project-roadmap.md](project-roadmap.md) — текущее направление и гейты;
- [validation-plan.md](validation-plan.md) — общая логика проверки;
- [glossary.md](glossary.md) — термины, сокращения и метрики;
- [metric-measurement.md](metric-measurement.md) — устройство метрической части.

Синтетический 3D-стенд:

- [synthetic-3d-gate-plan.md](synthetic-3d-gate-plan.md) — G0–G12;
- [synthetic-3d-smoke-report.md](synthetic-3d-smoke-report.md) — G0;
- [synthetic-3d-metric-recovery-report.md](synthetic-3d-metric-recovery-report.md) — G1;
- [synthetic-3d-height-sweep-report.md](synthetic-3d-height-sweep-report.md) — G2;
- [synthetic-3d-robustness-report.md](synthetic-3d-robustness-report.md) — G3–G6;
- [synthetic-3d-teach-repeat-report.md](synthetic-3d-teach-repeat-report.md) — G7;
- [synthetic-3d-annotation-robustness-report.md](synthetic-3d-annotation-robustness-report.md) — G8;
- [synthetic-3d-terrain-pose-plan.md](synthetic-3d-terrain-pose-plan.md) — план G9;
- [synthetic-3d-terrain-pose-report.md](synthetic-3d-terrain-pose-report.md) — отрицательный итог G9;
- [synthetic-3d-pose-interactions-report.md](synthetic-3d-pose-interactions-report.md) — взаимодействия G12;
- [synthetic-3d-texture-class-plan.md](synthetic-3d-texture-class-plan.md) — замороженный план G13;
- [synthetic-3d-texture-class-report.md](synthetic-3d-texture-class-report.md) — отрицательный результат G13;
- [synthetic-3d-texture-ambiguity-plan.md](synthetic-3d-texture-ambiguity-plan.md) — замороженный план G13-R;
- [synthetic-3d-texture-ambiguity-report.md](synthetic-3d-texture-ambiguity-report.md) — положительный результат G13-R;
- [synthetic-3d-lens-distortion-plan.md](synthetic-3d-lens-distortion-plan.md) — замороженный план G14-A;
- [synthetic-3d-lens-distortion-report.md](synthetic-3d-lens-distortion-report.md) — отрицательный raw-итог и положительная точная коррекция G14-A;
- [synthetic-3d-calibration-uncertainty-plan.md](synthetic-3d-calibration-uncertainty-plan.md) — замороженный план G14-A2;
- [synthetic-3d-calibration-uncertainty-report.md](synthetic-3d-calibration-uncertainty-report.md) — формально невалидный диагностический результат G14-A2;
- [synthetic-3d-calibration-uncertainty-repair-plan.md](synthetic-3d-calibration-uncertainty-repair-plan.md) — замороженный repair-план G14-A2-R;
- [synthetic-3d-calibration-uncertainty-repair-report.md](synthetic-3d-calibration-uncertainty-repair-report.md) — валидный отрицательный итог G14-A2-R;
- [synthetic-3d-rolling-shutter-plan.md](synthetic-3d-rolling-shutter-plan.md) — замороженный план G14-B;
- [synthetic-3d-rolling-shutter-repair-plan.md](synthetic-3d-rolling-shutter-repair-plan.md) — зарегистрированный численный repair G14-B-R;
- [synthetic-3d-rolling-shutter-report.md](synthetic-3d-rolling-shutter-report.md) — валидный отрицательный итог G14-B-R;
- [synthetic-3d-video-quality-plan.md](synthetic-3d-video-quality-plan.md) — замороженный план G14-C;
- [synthetic-3d-video-quality-repair-plan.md](synthetic-3d-video-quality-repair-plan.md) — зарегистрированный repair учёта G14-C-R;
- [synthetic-3d-video-quality-report.md](synthetic-3d-video-quality-report.md) — валидный отрицательный итог G14-C-R;
- [nerf-3dgs-prelesson.md](nerf-3dgs-prelesson.md) — проверенные камеры, типичные ошибки и вопросы к занятию;
- [3d-reconstruction-refrigerator.md](3d-reconstruction-refrigerator.md) — решение по отложенной 3D-ветке, ресурсы и точка возобновления;
- [synthetic-3d-pose-sweep-report.md](synthetic-3d-pose-sweep-report.md) — границы позы G11;
- [synthetic-3d-relative-displacement-report.md](synthetic-3d-relative-displacement-report.md) — относительный вектор промаха G9-R;
- [synthetic-3d-clutter-report.md](synthetic-3d-clutter-report.md) — статичный трёхмерный мусор G10;
- [blender-quickstart.md](blender-quickstart.md) — ручной просмотр сцены.

Реальные данные и поиск места:

- [ortholoc-real-data-spike.md](ortholoc-real-data-spike.md) — OrthoLoC;
- [cloud-poc-plan.md](cloud-poc-plan.md) — постановка CLOUD-проверки;
- [cloud-smoke-v2.md](cloud-smoke-v2.md) — известные Teach/Repeat-пары;
- [cloud-dinov2-vlad-smoke.md](cloud-dinov2-vlad-smoke.md) — поиск места;
- [cloud-sequence-retrieval-plan.md](cloud-sequence-retrieval-plan.md) и
  [cloud-sequence-retrieval-smoke.md](cloud-sequence-retrieval-smoke.md) —
  проверка последовательности;
- [cloud-end-to-end-gate-plan.md](cloud-end-to-end-gate-plan.md) — итоговый
  сквозной протокол и результат.

Инструменты и будущая платформа:

- [gradio-poc.md](gradio-poc.md) — интерфейс плоского прототипа;
- [hardware-requirements.md](hardware-requirements.md) — камера, синхронизация,
  телеметрия и вспомогательные датчики;
- [hardware-prototype-track.md](hardware-prototype-track.md) — паспорт
  имеющегося дрона, схема передней и надирной камер, питание, закупка и
  аппаратные гейты H0–H5.

Названия `*-plan.md` означают постановку и правила опыта. Названия
`*-report.md` или `*-smoke.md` содержат измеренный результат и его
интерпретацию. В старых документах постановка и результат иногда объединены;
это не означает, что критерий был назначен после просмотра данных.

## Тесты: `tests/`

Файлы `test_*.py` в основном повторяют имена модулей в
`src/aerial_mapper/`. Например, `tests/test_metric_recovery.py` проверяет
`metric_recovery.py`, а `tests/test_synthetic_3d.py` — оркестрацию
Blender-пайплайна без обязательного запуска Blender.

Дополнительные роли:

- `test_environment.py` — целостность окружения и импортов;
- `test_spatial_experiment.py` — свойства пространственной серии;
- `test_cloud_dataset.py` — чтение и согласование телеметрии CLOUD.

Тесты проверяют кодовые свойства и защитные отказы. Они не заменяют
экспериментальные метрики на отложенных сценах и реальных данных.

## Куда добавлять новое

- Повторно используемый алгоритм — в `src/aerial_mapper/`.
- Тест нового поведения — в `tests/`.
- Один исследовательский запуск — в `experiments/feasibility/`.
- Его фиксированные параметры — в `experiments/configs/`.
- План, методика или интерпретация — в `docs/`.
- Манифест внешних данных — в `data/manifests/`.
- Загружаемый набор — в игнорируемый подкаталог `data/`.
- Таблица, изображение, кэш или Blender-сцена — в `outputs/`.
- Установка или загрузка внешнего инструмента — в `scripts/`.

Если экспериментальная функция начинает копироваться, её следует перенести в
пакет. Если файл нельзя воспроизвести из кода, конфигурации и манифеста, его
нельзя считать надёжным результатом проекта.

## Частые команды

Создать окружение и проверить код:

```bash
uv sync
uv run pytest
uv run ruff check .
```

Запустить учебный интерфейс:

```bash
uv run python apps/poc_web.py
```

Повторить текущий G8:

```bash
uv run python experiments/feasibility/synthetic_3d_annotation_robustness.py
```

Открыть уже созданную сцену G8:

```bash
./scripts/open_blender_scene.sh outputs/synthetic_3d/g8_annotation_robustness/scene/scene.blend
```

Конкретные команды загрузки данных и запуска других опытов приведены в
соответствующих отчётах. Перед запуском тяжёлого эксперимента нужно проверить
его конфигурацию, каталог вывода и то, не будут ли перезаписаны нужные локальные
артефакты.
