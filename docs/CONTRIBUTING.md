# Contributing

## Ветки

| Ветка | Назначение |
| --- | --- |
| `main` | Стабильная версия. Отсюда ставятся теги релизов |
| `deploy` | Интеграция и проверка перед попаданием в `main` |
| `feat/...`, `fix/...`, `refactor/...`, `chore/...`, `docs/...` | Рабочие ветки под одну задачу |

Напрямую в `deploy` и `main` не коммитим.

## Порядок работы

1. Создать рабочую ветку от актуального `deploy`:

   ```bash
   git switch deploy
   git pull
   git switch -c feat/short-description
   ```

2. Сделать изменения и закоммитить их.
3. Слить рабочую ветку в `deploy` и проверить там: CI, запуск приложения, нужные сценарии.
4. Когда `deploy` проверен, открыть PR из `deploy` в `main`.

## Перед коммитом

Установить зависимости и pre-commit-хуки:

```bash
uv sync --group experiments --group rl-cpu
uv run pre-commit install
```

Проверки:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Коммиты

Используем [Conventional Commits](https://www.conventionalcommits.org/ru/):
`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.

Один коммит — одно логическое изменение. Рефакторинг и новую функциональность
коммитим отдельно.

## Релиз

1. Поднять версию в `pyproject.toml` отдельным коммитом: `chore: bump version to X.Y.Z`.
2. После слияния в `main` поставить тег `vX.Y.Z` на `main`.
3. По тегу CD собирает и публикует Docker-образ. Тег должен совпадать с версией в
   `pyproject.toml`, иначе сборка остановится.
