"""Dump every string that reaches a model prompt, so two code trees can be compared.

Covers:
- the module-level DSPy Signatures (instructions, field prefixes and descriptions);
- the agent Signature as the ReAct pipeline builds it at run time, in all four
  evaluation modes (4D or 1D scoring, with or without the evaluate tool);
- the Signatures defined inside functions (the LLM judge and the faithfulness
  check in evaluation/metrics.py), as their source with comments dropped;
- the docstrings of the tools the ReAct agent receives (both evaluation modes);
- the string constants of config/prompts.py and pipeline/manual.py;
- the functions that format passages into a prompt (pipeline/base.py,
  pipeline/crag.py) and the FinanceBench section-title prompt
  (scripts/build_index.py), as their source with docstrings and comments dropped.

Writes sorted JSON to stdout. No API call.

Not covered: the JSON the agent's tools return as observations, and the text
DSPy and litellm add around these strings (fixed by uv.lock).

Usage:
    uv run python scripts/dump_prompts.py [code-tree] > prompts.json

code-tree defaults to this repository. To compare with the code the runs were
made from, dump that tree too and diff the two files:

    uv run python scripts/dump_prompts.py /path/to/run-code > run.json
    uv run python scripts/dump_prompts.py > here.json
    diff run.json here.json
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import pkgutil
import sys
from pathlib import Path

# Functions whose code builds prompt text: (file, class or None, function).
PROMPT_CODE = [
    ("agentic_rag/pipeline/base.py", "BasePipeline", "format_passages"),
    ("agentic_rag/pipeline/crag.py", "CRAGReplicaPipeline", "_refine_passages"),
    ("scripts/build_index.py", None, "_extract_section_titles_llm"),
]


def _signature_dump(sig) -> dict[str, object]:
    fields = {
        k: [str(v.json_schema_extra.get("prefix")), str(v.json_schema_extra.get("desc"))]
        for k, v in sig.model_fields.items()
    }
    return {"instructions": sig.instructions, "fields": fields}


def _strip_docstrings(node: ast.AST) -> ast.AST:
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                n.body = n.body[1:] or [ast.Pass()]
    return node


def _nested_signatures(root: Path) -> dict[str, str]:
    """Signature classes defined inside a function, as source without comments."""
    out: dict[str, str] = {}
    for path in sorted((root / "agentic_rag").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.ClassDef) and any(
                    "Signature" in ast.unparse(b) for b in node.bases
                ):
                    key = f"NESTED_SIG {path.relative_to(root)}:{func.name}.{node.name}"
                    out[key] = ast.unparse(node)
    return out


def _prompt_code(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel, cls, name in PROMPT_CODE:
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        scope = tree.body
        if cls is not None:
            scope = next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        func = next((n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name), None)
        key = f"CODE {rel}:{cls + '.' if cls else ''}{name}"
        out[key] = ast.unparse(_strip_docstrings(func)) if func else "MISSING"
    return out


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
    root = root.resolve()
    sys.path.insert(0, str(root))
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
                out[f"SIG {m.name}.{attr}"] = _signature_dump(obj)
            elif isinstance(obj, str) and not attr.startswith("__"):
                out[f"STR {m.name}.{attr}"] = obj

    from agentic_rag.signatures.agent import make_agent_signature

    for four_d in (True, False):
        for has_evaluate in (True, False):
            sig = make_agent_signature(enable_4d=four_d, has_evaluate=has_evaluate)
            mode = f"{'4d' if four_d else '1d'}:{'eval' if has_evaluate else 'noeval'}"
            out[f"AGENT_SIG {mode}"] = _signature_dump(sig)

    out.update(_nested_signatures(root))
    out.update(_prompt_code(root))

    from agentic_rag.config.settings import settings
    from agentic_rag.tools import TOOL_REGISTRY, create_tools

    for four_d in (True, False):
        settings.experiment.enable_4d_evaluation = four_d
        for tool in create_tools(None, None, None, enabled_tools=list(TOOL_REGISTRY)):
            out[f"TOOL {'4d' if four_d else '1d'}:{tool.__name__}"] = tool.__doc__

    print(json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
