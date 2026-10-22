import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import mlflow

from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.task_pipeline.schemas import HexState


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
        "generator": "scripts.backpressure.make_dataset:make_dataset",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate inference inputs for a pinned MLflow model"
    )
    parser.add_argument("--model-uri", required=True, help="models:/<name>/<version>, not an alias")
    parser.add_argument("--tracking-uri", default="http://mlflow:5000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--invalid-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if not args.model_uri.startswith("models:/") or not args.model_uri.rsplit("/", 1)[-1].isdigit():
        parser.error("model-uri must pin a numeric model version")
    mlflow.set_tracking_uri(args.tracking_uri)
    info = mlflow.models.get_model_info(args.model_uri)
    if info.metadata is None or "encoder" not in info.metadata:
        raise ValueError("Model has no encoder metadata")
    manifest = make_dataset(
        ObservationEncoder.from_config(info.metadata["encoder"]),
        args.output,
        size=args.size,
        seed=args.seed,
        invalid_fraction=args.invalid_fraction,
        model_uri=args.model_uri,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
