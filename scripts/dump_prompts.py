"""Dump every string that reaches a model prompt, so two code trees can be compared.

Covers the DSPy Signatures (instructions, field prefixes and descriptions), the
docstrings of the tools the ReAct agent receives (both evaluation modes), and
the string constants of config/prompts.py and pipeline/manual.py. Writes
sorted JSON to stdout. No API call.

Usage:
    uv run python scripts/dump_prompts.py [code-tree] > prompts.json

code-tree defaults to this repository. To compare with the code the runs were
made from, dump that tree too and diff the two files:

    uv run python scripts/dump_prompts.py /path/to/run-code > run.json
    uv run python scripts/dump_prompts.py > here.json
    diff run.json here.json
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import pkgutil
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
    sys.path.insert(0, str(root.resolve()))
    os.chdir(root)

    import dspy

    import agentic_rag
    import agentic_rag.pipeline  # imported before agentic_rag.tools

    out: dict[str, object] = {}
    for m in pkgutil.walk_packages(agentic_rag.__path__, "agentic_rag."):
        try:
            mod = importlib.import_module(m.name)
        except Exception as e:  # report, do not hide, a module that fails to import
            out[f"IMPORT_FAIL {m.name}"] = repr(e)[:200]
            continue
        for attr, obj in vars(mod).items():
            prompt_constant = isinstance(obj, str) and m.name.endswith(
                ("config.prompts", "pipeline.manual")
            )
            if getattr(obj, "__module__", None) != mod.__name__ and not prompt_constant:
                continue
            if (
                inspect.isclass(obj)
                and issubclass(obj, dspy.Signature)
                and obj is not dspy.Signature
            ):
                fields = {
                    k: [
                        str(v.json_schema_extra.get("prefix")),
                        str(v.json_schema_extra.get("desc")),
                    ]
                    for k, v in obj.model_fields.items()
                }
                out[f"SIG {m.name}.{attr}"] = {"instructions": obj.instructions, "fields": fields}
            elif isinstance(obj, str) and not attr.startswith("__"):
                out[f"STR {m.name}.{attr}"] = obj

    from agentic_rag.config.settings import settings
    from agentic_rag.tools import TOOL_REGISTRY, create_tools

    for four_d in (True, False):
        settings.experiment.enable_4d_evaluation = four_d
        for tool in create_tools(None, None, None, enabled_tools=list(TOOL_REGISTRY)):
            out[f"TOOL {'4d' if four_d else '1d'}:{tool.__name__}"] = tool.__doc__

    print(json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
