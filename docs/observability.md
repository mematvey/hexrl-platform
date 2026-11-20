# Наблюдаемость конвейера

## Состав и профили Compose

`docker-compose.yaml` подключает `observability/compose.yaml` через `include`.
Обычный `docker compose up -d` запускает приложение без Grafana, хранилищ и
Collector. Профили включают стек по частям:

| Профиль | Компоненты |
| --- | --- |
| `logs` | Alloy и Loki |
| `traces` | Collector для трейсов и Tempo |
| `metrics` | Collector для метрик и Prometheus |
| `observability` | Все компоненты, единый Collector, Grafana |

```powershell
docker compose up -d
docker compose --profile logs up -d
docker compose --profile traces up -d
docker compose --profile metrics up -d
docker compose --profile traces --profile metrics up -d
docker compose --profile observability up -d
```

Trace-only и metrics-only профили используют отдельные конфигурации Collector,
чтобы не отправлять сигнал в отсутствующее хранилище. Профиль `observability`
использует конфигурацию с обоими pipeline. Профили `traces` и `metrics` можно
запускать вместе: сигналы отправляются по отдельным
OTLP endpoints. В профиле `observability` эти endpoints ведут к одному Collector.
Приложения не зависят через `depends_on` от стека наблюдаемости.

Для экспорта приложение должно получить `OTEL_SDK_DISABLED=false`; по умолчанию
SDK отключён, поэтому обычный запуск не зависит от Collector. Для профиля
`traces` оставить `OTEL_METRICS_EXPORTER=none`, а для `metrics` —
`OTEL_TRACES_EXPORTER=none`. В полном профиле включить оба экспортёра. Переменные
читаются процессом при старте, поэтому после изменения окружения пересоздать
приложения. В Compose traces отправляются на `otel-collector-traces:4318`, а
metrics — на `otel-collector-metrics:4318`. В полном профиле оба имени указывают
на combined Collector.

## Логи: Alloy → Loki

Alloy через Docker API читает stdout контейнеров текущего проекта, отобранных по
`com.docker.compose.project`; затем оставляет `app`, `worker` и `batch-source`.
В Loki индексируются только `service`, `level` и `env`. Идентификаторы задач и
трейсов остаются в JSON-строке, чтобы число потоков логов не росло с каждой
задачей. Сохраняются исходная запись приложения и её время. Позиция чтения Alloy
и данные Loki хранятся в именованных томах. Ротация Docker JSON-логов задана в
Compose.

Для короткоживущего `batch-source` Compose-команда оставляет контейнер работающим
на `BATCH_SOURCE_LOG_GRACE_S` до обработки и после неё. Паузы увеличивают окно
обнаружения и чтения, но не дают абсолютной гарантии доставки и требуют проверки
на живом стенде. Не запускать этот сервис с `--rm`: Docker должен сохранить
контейнер и его stdout для агента.

Пример запроса в Grafana Explore:

```logql
{service="hexrl-api"} | json | event="http_request_completed"
```

Поиск по задаче: добавить `| json | task_id="..."`. `docker.sock:ro` задаёт
read-only mount, но не ограничивает операции, разрешённые Docker API.

## Трейсы: OpenTelemetry → Collector → Tempo

SDK отправляет OTLP HTTP/protobuf на Collector. Он применяет `memory_limiter`
перед `batch`, ограничивает очередь и повторы экспорта, затем направляет спаны в
Tempo. У API, source и worker отдельные `service.name`; hostname контейнера
различает экземпляры воркера. Полный сбор трейсов выбран для учебной нагрузки.

Обычный Tempo config работает без Prometheus. Service Graph — дополнительный
режим из предыдущего этапа: он включается `TEMPO_CONFIG_FILE=service-graph.yaml`
только с профилем `observability`, где доступен Prometheus. Для standalone
`traces` оставить `TEMPO_CONFIG_FILE=config.yaml`.

## Метрики: OTLP и Prometheus

Приложения экспортируют метрики по OTLP через Collector в Prometheus OTLP
receiver. Для короткоживущего source push выбран, чтобы метрики успевали попасть
в хранилище даже без pull-опроса самого процесса. Глубину очереди и число
unacked сообщений Prometheus получает scrape-ом встроенного RabbitMQ
Prometheus-плагина на порту 15692. `QUEUE_PREFIX` ограничивает scrape очередями
текущего конвейера (`tasks`, `retry`, `dlq`), чтобы старые экспериментальные
префиксы не разрастались в Prometheus.

При OTLP-приёме Prometheus использует ограниченный набор resource attributes:
`service.name`, `service.instance.id`, `service.version` и
`deployment.environment.name`; произвольные resource attributes не продвигаются
в labels. Метрика лимита помечена именем очереди. Для неограниченной очереди
точка не экспортируется, поэтому линия лимита и алерт заполнения отсутствуют.

Оценка кардинальности, контракт метрик и причины выбора границ histogram
описаны ниже в разделе «Метрики»; это оценки, не измерение runtime-серий.

## Grafana provisioning и сигналы

Datasource UID заданы в provisioning: `metrics`, `logs`, `traces`. Dashboard,
правила алертов и источники находятся в репозитории и загружаются при старте
Grafana. Dashboard содержит выбор сервиса, экземпляра worker и очереди; очередь
для выбора берётся из метрики настроенного лимита. Перцентили вычисляются из
объединённых histogram buckets, счётчики отображаются через `rate` или
`increase`, панели задают единицы.

В Loki настроено derived field из JSON `trace_id` в Tempo. В Tempo настроен
запрос к Loki по тому же trace ID с временным запасом. Очередной алерт вычисляет
`ready / configured_limit` отдельно для каждой ограниченной очереди и срабатывает
при значении выше 0.9 в течение минуты. DLQ alert срабатывает, пока в DLQ есть
сообщения, с выдержкой 10 секунд. NoData и ошибки запроса не считаются здоровым
состоянием. Наличие правила в provisioning само по себе не подтверждает его
срабатывание; для этого нужен сценарий на живом стенде.

## Ресурсы и известные ограничения проверки

Заданные Compose memory limits: Loki 256 MiB, Tempo 256 MiB, Prometheus 256 MiB,
Collector 192 MiB, Grafana 192 MiB, Alloy 128 MiB. Суммарный лимит — 1280 MiB.
Это верхняя конфигурационная сумма, а не измеренное потребление. Для подтверждения
бюджета нужен `docker stats` под нагрузкой.

Версии образов закреплены, retention хранилищ по умолчанию 24 часа, данные и
позиции чтения находятся в именованных томах. Отсутствие Collector или хранилища
не является зависимостью приложения; SDK отключён по умолчанию. При включённом
экспорте краткий экспортный таймаут и ограниченные очереди уменьшают влияние
сбоев телеметрии, но при длительной недоступности backend часть телеметрии может
быть потеряна.
## Настройка

По умолчанию `OTEL_SDK_DISABLED=true`: экспортёры и их фоновые потоки не создаются.
Логи остаются JSON в stdout. API, источник и воркер имеют отдельные имена сервисов.
Для экспорта трейсов и метрик в доступный OpenTelemetry Collector задать в окружении процесса
или в env-файле контейнера:

```dotenv
OTEL_SDK_DISABLED=false
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
OTEL_EXPORTER_OTLP_TIMEOUT=0.5
OTEL_TRACES_SAMPLER=parentbased_always_on
OTEL_TRACES_EXPORTER=otlp
OTEL_METRICS_EXPORTER=otlp
OTEL_METRIC_EXPORT_INTERVAL=5000
OTEL_METRIC_EXPORT_TIMEOUT=1000
OTEL_RESOURCE_ATTRIBUTES=deployment.environment.name=development
LOG_LEVEL=INFO
```

Этот этап использует OTLP HTTP/protobuf. Адрес Collector должен разрешаться из
процесса отправителя: для запуска Python на хосте обычно нужен `localhost`,
для контейнера — сетевое имя Collector. Хранилища приложению неизвестны.
При запущенном профиле наблюдаемости имя `otel-collector` разрешается Compose-сетью.
`OTEL_TRACES_EXPORTER=none` или `OTEL_METRICS_EXPORTER=none` отключают отдельный
сигнал. `OTEL_EXPORTER_OTLP_TIMEOUT` — сетевой таймаут в секундах;
`OTEL_METRIC_EXPORT_INTERVAL` и `OTEL_METRIC_EXPORT_TIMEOUT` — миллисекунды.

`OTEL_SERVICE_NAME` переопределяет имя сервиса. Не задавать одно общее имя всем
сервисам: значения по умолчанию — `hexrl-api`, `hexrl-source`, `hexrl-worker`.
`service.version`, `deployment.environment.name` и `service.instance.id` можно
переопределить через `OTEL_RESOURCE_ATTRIBUTES`. По умолчанию версия берётся из
пакета, instance ID — hostname контейнера, поэтому реплики воркера различаются.

Переменные OTEL читаются из окружения. При локальном запуске `uv run` их нужно
экспортировать в PowerShell через `$env:...`; загрузка прикладных Settings из `.env`
не экспортирует их в окружение SDK. В Compose переменные поступают через `env_file`.

## Путь задачи

HTTP создаёт серверный спан. `task_id` передаётся в заголовке RabbitMQ и в
`correlation_id`, W3C `traceparent` переносит контекст. Стандартная
`RabbitTelemetryMiddleware` создаёт спаны публикации и обработки. Для batch source
каждая строка начинает отдельный `source.task`, а `batch_id` связывает пакет.
Весь файл не помещается в один трейс.

Ручные спаны:

- `db.create_task` — фиксация приёма HTTP-задачи;
- `source.wait` — пауза после отказа брокера;
- `inference` — чистый predict с `model.alias` и `model.version`;
- `db.save_result` — сохранение результата вместе с commit.

Искусственная задержка остаётся внутри обработки, но вне `inference`: на трейсе
можно отличить стоимость настоящей модели от имитации нагрузки. `copy_context()`
переносит OTel и контекст логов в каждый вызов ThreadPoolExecutor отдельно.
Номер попытки хранится в `messaging.delivery.attempt`, аварийная повторная доставка
отмечается атрибутом `messaging.message.redelivered`.

Повторы сохраняют trace ID через retry и TTL-пересылку. Ошибки отмечаются ERROR и
событием exception, перевод в DLQ — событием `task.dlq`. 503 также имеет ERROR-трейс.
Интервал между публикацией и обработкой виден по временной шкале; при измерении
его длительности необходимо учитывать синхронизацию часов разных процессов.

## Логи и защита данных

`task_context()` привязывает идентификаторы к текущему обработчику и очищает их
даже при исключении. Форматтер автоматически добавляет `task_id`, `batch_id`,
`trace_id`, `span_id`, когда соответствующий контекст существует. HTTP-запись об
ответе также содержит идентификатор созданной задачи, в том числе при 503.

Ошибки сохраняют тип исключения и кадры стека без исходной записи и значений
аргументов. Текст строк исходного кода в стек не включается: там тоже могут быть
чувствительные литералы. DSN и значения секретов из переменных окружения
маскируются. Не следует добавлять payload, конфиги или произвольный текст исключения
в события приложения. Успешные `/healthz` записываются только на DEBUG.

События включают загрузку модели с alias и версией, начало/конец обработки,
отказ публикации, backoff источника, ошибку с попыткой, DLQ и graceful shutdown.
Аналитические CSV старых нагрузочных скриптов пока сохранены для совместимости;
они не заменяют логи и не используются как хранилище телеметрии.

## Отказы телеметрии

Экспорт выполняется BatchSpanProcessor в отдельном потоке: очередь ограничена
128 спанами, размер пакета — до 128, интервал — 1 секунда. При переполнении спаны
теряются, а обработка задач продолжается. Экспорт ограничен коротким таймаутом.
При нормальном завершении SDK закрывается после завершения обработки и работы
с брокером; у короткоживущего источника также выполняется финальный экспорт.
После SIGKILL незавершённые спаны и буфер могут потеряться.

На стенде используется parent-based always-on, чтобы находить произвольные задачи.
При внешнем unsampled traceparent решение родителя сохраняется.
Tail sampling не используется.

## Метрики

Приложения отправляют метрики по OTLP HTTP/protobuf через Collector. Это важно
для короткоживущего источника: pull может не успеть его опросить. В начале источника
экспортируется нулевая точка, при нормальном завершении — финальные накопленные
значения. Оба вызова выполняются вне event loop с ограничением времени экспорта.
Периодический экспорт идёт в потоке SDK; обработчики не выполняют сетевые запросы
при увеличении счётчика. Используется cumulative temporality для Prometheus.

Контракт ниже использует имена OTLP. Prometheus добавляет суффиксы единиц и типов;
отображение имён зависит от конфигурации приёмника.

| Метрика | Тип и единица | Labels | Что измеряется |
| --- | --- | --- | --- |
| `hexrl.api.requests` | counter, 1 | method, route, status_code | Все ответы API, кроме healthz, включая 422, 503 и необработанные 500 |
| `hexrl.api.duration` | histogram, s | method, route, status_code | Длительность обработки HTTP-запроса |
| `hexrl.source.published` | counter, 1 | нет | Успешные публикации после confirm |
| `hexrl.source.rejections` | counter, 1 | reason | nack, timeout и unavailable отдельно |
| `hexrl.source.wait` | counter, s | нет | Фактически проведённое в backoff время, включая прерванную паузу |
| `hexrl.source.invalid` | counter, 1 | нет | Невалидные строки файла |
| `hexrl.worker.attempts` | counter, 1 | outcome | Исходы доставок: ok, error, dlq, duplicate, cancelled |
| `hexrl.worker.completed` | counter, 1 | нет | Новые результаты после commit, без дубликатов |
| `hexrl.worker.redeliveries` | counter, 1 | нет | Доставки с AMQP redelivered=true |
| `hexrl.worker.retries` | counter, 1 | нет | Подтверждённые публикации в retry |
| `hexrl.worker.dlq.publications` | counter, 1 | нет | Подтверждённые публикации в DLQ |
| `hexrl.worker.inflight` | gauge, 1 | нет | Активные обработчики сообщений |
| `hexrl.worker.predict.duration` | histogram, s | outcome | Чистый predict, включая ошибки |
| `hexrl.worker.delay.duration` | histogram, s | нет | Искусственная задержка, отдельно от predict |
| `hexrl.worker.processing.duration` | histogram, s | outcome | Время доставки в обработчике до выхода, с БД и подтверждениями |
| `hexrl.worker.task.duration` | histogram, s | нет | От submitted_at до фиксации нового успешного результата |
| `hexrl.worker.invalid.timestamps` | counter, 1 | нет | Отрицательная задержка или submitted_at без часового пояса |
| `hexrl.process.memory` | gauge, By | нет | RSS текущего процесса из psutil |
| `hexrl.process.running` | gauge, 1 | нет | 1 при работе, 0 в финальном экспорте перед остановкой SDK |
| `hexrl.queue.limit` | gauge, 1 | queue | Настроенный лимит у API/источника; для unbounded точка отсутствует |

Одна доставка может завершиться ошибкой и назначить retry: это увеличит attempts
с outcome=error и retries, но не completed. TTL-пересылка создаёт новую публикацию
и сама по себе не означает AMQP redelivered. Повторная публикация в DLQ считается
как публикация, а не как новая уникальная FAILED-задача.

`save_result()` уже возвращает признак новой вставки. Счётчик completed и histogram
task.duration обновляются только после commit такой вставки. Два параллельных
вычисления одной задачи дают один новый результат и две попытки. Метрики не атомарны
с БД: падение между commit и увеличением counter может недосчитать результат;
достоверная сверка уникальных задач по-прежнему выполняется по PostgreSQL.

In-flight уменьшается в finally при ошибке или отмене. Он показывает обработчики,
а не число занятых потоков: отмена await не прерывает уже выполняющийся predict.
Сама модель может закончить позже; shutdown пула ожидает её завершения.

Старый CSV source.wait_s_total сохраняет сумму назначенных пауз для совместимости.
Новая OTLP-метрика source.wait измеряет реальное время монотонными часами.
Timeout подтверждения не доказывает, что брокер отверг публикацию; reason=nack
выделяет настоящий отрицательный ответ отдельно от сетевых сбоев.

API использует разрешённый метод и шаблон сопоставленного маршрута. Неизвестные
адреса объединены в route=unmatched, неизвестные методы — method=OTHER.
task_id, batch_id, trace_id, версии моделей и текст ошибок в labels не добавляются.
Views дополнительно отбрасывают неожиданные ключи атрибутов. Автоматические метрики
FastStream отключены через DropAggregation; его трассировка остаётся включённой.

### Гистограммы и задержки

Predict имеет границы от 1 мс с детальными бакетами вокруг измеренных 2–3 мс.
HTTP имеет отдельные бакеты около дедлайнов 0.5/0.75/1 секунды. Для processing и
delay добавляются границы 0.5×, 1×, 1.1×, 2× и 4× INFERENCE_DELAY_S. Для долгих
проверок сохранены границы до 120 с. Полная задержка имеет бакеты до 300 с,
включая 10/15/30/60/120 с. Настройки применяются при запуске SDK.

Predict, delay и processing измеряются монотонными часами. Task duration связывает
время разных процессов и требует синхронизации часов. Отрицательные значения
не заменяются нулями: они пропускаются с увеличением invalid.timestamps.
Положительное значение само по себе не доказывает правильность синхронизации.
Histogram task.duration описывает только успешные задачи: проверка SLO должна
отдельно учитывать ошибки и выбранное окно/множество задач.

### Память и завершение источника

После нормального выхода source.running=0 позволяет не показывать последнее RSS
как память всё ещё работающего источника. После SIGKILL финального значения нет,
поэтому актуальность определяется свежестью последнего измерения. Push-модель
не предоставляет автоматически эквивалент Prometheus up для жизнеспособности процесса.
Приложение не публикует собственных оценок глубины очередей: источник
показателей ready/unacked/DLQ — встроенный Prometheus-плагин RabbitMQ.

### Оценка числа рядов

Для классической Prometheus histogram с K конечными границами получается K+3
ряда на сочетание labels: K бакетов, +Inf, sum и count. API имеет 12 конечных
границ: requests и duration вместе дают 16 рядов на сочетание method/route/status.
Если A — число сочетаний, верхняя оценка API — 16A+3 с runtime и queue.limit.
Грубая оценка A — произведение числа методов, маршрутов и статусов, а не число UUID.

При delay=0.5 один worker имеет до 183 рядов приложения после появления всех
допустимых исходов: 34 predict, 20 delay, 100 processing, 16 task duration,
10 counters и 3 gauges. Source — до 9 рядов при bounded queue. Например, при
A=10 и двух воркерах: 163+366+9=538 рядов. Это оценка инструментов приложения;
target_info, RabbitMQ и собственные метрики хранилищ считаются дополнительно.
Реплики и перезапуски с новым service.instance.id увеличивают число рядов за retention.

## Проверки

```powershell
uv sync --frozen --group experiments --group rl-cpu --group inference
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check
uv run --no-sync pytest
uv run --no-sync pytest -m integration tests/integration
```

Обычные тесты проверяют изоляцию параллельных задач, родителя inference в потоке,
очистку контекста после ошибки, отдельные трейсы строк файла, backoff, маскировку,
реальный OTLP HTTP/protobuf экспорт и работу без Collector. Интеграционные тесты
дополнительно проверяют путь HTTP → настоящий RabbitMQ → поток → БД, а также
сохранение traceparent через настоящую retry-очередь до DLQ, включая сбой публикации.
