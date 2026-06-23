import logging
from datetime import date
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def load_registry():
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def save_registry(registry):
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)


def register_model(
    name,
    model_path,
    model_type="finetuned",
    base_model=None,
    dataset=None,
    classes=None,
    num_images=None,
    metrics=None,
    class_metrics=None,
    notes=None,
):
    registry = load_registry()

    entry = {
        "type": model_type,
        "date": str(date.today()),
        "path": str(model_path),
        "classes": classes or [],
        "num_classes": len(classes) if classes else 0,
    }
    if base_model:
        entry["base_model"] = base_model
    if dataset:
        entry["dataset"] = dataset
    if num_images:
        entry["num_images"] = num_images
    if metrics:
        entry["metrics"] = metrics
    if class_metrics:
        entry["class_metrics"] = class_metrics
    if notes:
        entry["notes"] = notes

    registry[name] = entry
    save_registry(registry)
    log.info("Registered model '%s' in %s", name, REGISTRY_PATH)
    return entry


def get_model(name):
    registry = load_registry()
    entry = registry.get(name)
    if entry is None:
        raise KeyError(f"Model '{name}' not found in registry. Available: {list(registry.keys())}")
    return entry


def list_models(model_type=None):
    registry = load_registry()
    if model_type:
        return {k: v for k, v in registry.items() if v.get("type") == model_type}
    return registry


def resolve_model_path(name_or_path):
    p = Path(name_or_path)
    if p.exists():
        return str(p.resolve())
    try:
        entry = get_model(name_or_path)
        return str(Path(entry["path"]).resolve())
    except KeyError:
        raise ValueError(
            f"'{name_or_path}' is neither an existing file path nor a registered model name. "
            f"Available models: {list(list_models().keys())}"
        )
