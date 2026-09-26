"""Run-scoped artifact paths; BITFI_RUN_ROOT isolates complete reproductions."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def experiment_path(relative, legacy):
    configured = os.environ.get("BITFI_RUN_ROOT")
    return Path(configured).resolve() / relative if configured else ROOT / legacy

def selected_context(family):
    import json
    return int(json.loads(experiment_path("context_validation/selected_contexts.json", "03_result/context_validation_20260925/selected_contexts.json").read_text())[family])
