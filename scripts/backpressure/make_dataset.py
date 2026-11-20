import argparse
import json
from pathlib import Path
from typing import Any

import mlflow

from hexrl_platform.rl import inputs
from hexrl_platform.rl.observation import ObservationEncoder


def make_dataset(
    encoder: ObservationEncoder,
    output: Path,
    *,
    size: int,
    seed: int,
    invalid_fraction: float = 0.0,
    model_uri: str,
) -> dict[str, Any]:
    """Сохраняет прежний API скрипта и передаёт генерацию общему RL-модулю."""
    return inputs.make_dataset(
        encoder,
        output,
        size=size,
        seed=seed,
        invalid_fraction=invalid_fraction,
        model_uri=model_uri,
    )


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
