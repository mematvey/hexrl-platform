import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.task_pipeline.schemas import HexState


@dataclass(frozen=True, slots=True)
class InputSpec:
    name: str
    size: int
    seed: int
    invalid_fraction: float = 0.0


# smoke - короткий прогон для разработки, defense - длинный пакет для защиты,
# invalid - состояния вне карты: постоянная доменная ошибка без повторов
DEFAULT_INPUTS = (
    InputSpec("smoke", 1000, 42),
    InputSpec("defense", 6000, 43),
    InputSpec("invalid", 20, 44, invalid_fraction=1.0),
)


def make_dataset(
    encoder: ObservationEncoder,
    output: Path,
    *,
    size: int,
    seed: int,
    invalid_fraction: float = 0.0,
    model_uri: str,
) -> dict[str, Any]:
    if size < 1 or not 0 <= invalid_fraction <= 1:
        raise ValueError("size must be positive and invalid_fraction must be between 0 and 1")
    rng = random.Random(seed)
    invalid = set(random.Random(seed).sample(range(size), round(size * invalid_fraction)))
    cells = encoder.hex_map.cells
    digest = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        for index in range(size):
            start, goal = rng.sample(cells, 2)
            state = HexState(
                q=start.q,
                r=start.r,
                goal_q=goal.q,
                goal_r=goal.r,
                steps_remaining=rng.randint(1, encoder.max_steps),
            )
            if index in invalid:
                state = state.model_copy(
                    update={"q": encoder.hex_map.radius + 1, "r": encoder.hex_map.radius + 1}
                )
            line = (json.dumps(state.model_dump(), sort_keys=True) + "\n").encode()
            stream.write(line)
            digest.update(line)
    manifest = {
        "generator": "hexrl_platform.rl.inputs:make_dataset",
        "rows": size,
        "seed": seed,
        "invalid_rows": len(invalid),
        "sha256": digest.hexdigest(),
        "model_uri": model_uri,
        "encoder": encoder.to_config(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _is_current(path: Path, spec: InputSpec, model_uri: str) -> bool:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return (
        manifest.get("model_uri") == model_uri
        and manifest.get("rows") == spec.size
        and manifest.get("seed") == spec.seed
        and manifest.get("invalid_rows") == round(spec.size * spec.invalid_fraction)
        and manifest.get("sha256") == digest
    )


def ensure_inputs(
    encoder: ObservationEncoder,
    directory: Path,
    model_uri: str,
    specs: Sequence[InputSpec] = DEFAULT_INPUTS,
) -> dict[str, bool]:
    """Генерирует только отсутствующие или устаревшие файлы; возвращает, что пересоздано."""
    generated = {}
    for spec in specs:
        path = directory / f"{spec.name}.jsonl"
        generated[spec.name] = not _is_current(path, spec, model_uri)
        if generated[spec.name]:
            make_dataset(
                encoder,
                path,
                size=spec.size,
                seed=spec.seed,
                invalid_fraction=spec.invalid_fraction,
                model_uri=model_uri,
            )
    return generated
