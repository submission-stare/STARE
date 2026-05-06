from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
RISK_CONTROL_PATTERNS = (
    "margin",
    "liquidat",
    "stop_loss",
    "stop loss",
    "borrow",
    "leverage_cap",
    "max_leverage",
)

IGNORED_CODE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "site-packages",
    "tests",
}

SUPPLEMENTAL_CODE_FILENAMES = {
    "run.py",
    "run_st.py",
    "run_st_strategist.py",
    "continue_llm_trading.py",
    "post_processing.py",
    "upload_results_to_s3.py",
}
SUPPLEMENTAL_CONFIG_FILENAMES = {"config.yaml", "config.yml"}
SUPPLEMENTAL_EXPERIMENT_ROOT = ("experiments", "liu_et_al_2020")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def _module_name_for_path(path: str, scope_root: str) -> str:
    rel = os.path.relpath(path, scope_root)
    no_suffix = rel[:-3] if rel.endswith(".py") else rel
    parts = no_suffix.split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(part for part in parts if part)


def _build_module_index(scope_root: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for root, dirs, files in os.walk(scope_root):
        dirs[:] = [
            dirname
            for dirname in dirs
            if dirname not in IGNORED_CODE_DIRS
        ]
        for filename in files:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(root, filename)
            module = _module_name_for_path(path, scope_root)
            if module:
                index[module] = path
    return index


def _virtual_module_name(path: str, scope_root: str) -> str:
    rel = os.path.relpath(path, scope_root)
    no_suffix = rel[:-3] if rel.endswith(".py") else rel
    return ".".join(part for part in no_suffix.split(os.sep) if part)


def _candidate_result_roots(data_path: str) -> list[str]:
    if not data_path:
        sibling = os.path.abspath(os.path.join(os.getcwd(), "..", "results_to_explain"))
        return _candidate_result_roots(sibling) if os.path.isdir(sibling) else []

    path = os.path.abspath(data_path)
    if os.path.basename(path) == "benchmark-data":
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            return [parent]
    if os.path.isdir(os.path.join(path, "benchmark-data")) or any(
        os.path.exists(os.path.join(path, filename))
        for filename in SUPPLEMENTAL_CODE_FILENAMES | SUPPLEMENTAL_CONFIG_FILENAMES
    ):
        return [path]
    if os.path.isdir(path):
        return [
            os.path.join(path, name)
            for name in sorted(os.listdir(path))
            if os.path.isdir(os.path.join(path, name))
        ]
    return []


def discover_supplemental_code_placeholders(
    *,
    data_path: str = "",
    code_scope_root: str,
) -> list[dict[str, Any]]:
    """Return result-run source snapshots as virtual files under the code root."""
    scope_root = os.path.abspath(code_scope_root)
    placeholders: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for result_root in _candidate_result_roots(data_path):
        if not os.path.isdir(result_root):
            continue
        run_label = os.path.basename(result_root.rstrip(os.sep))
        for filename in sorted(SUPPLEMENTAL_CODE_FILENAMES | SUPPLEMENTAL_CONFIG_FILENAMES):
            source_path = os.path.join(result_root, filename)
            if not os.path.isfile(source_path):
                continue
            try:
                source = _read_text(source_path)
            except OSError:
                continue
            digest = hashlib.sha1(source.encode("utf-8", errors="replace")).hexdigest()
            key = (run_label, filename)
            if key in seen:
                continue
            seen.add(key)
            virtual_path = os.path.join(scope_root, *SUPPLEMENTAL_EXPERIMENT_ROOT, run_label, filename)
            placeholders.append(
                {
                    "path": os.path.abspath(virtual_path),
                    "source_path": os.path.abspath(source_path),
                    "relative_path": os.path.relpath(virtual_path, os.getcwd()),
                    "source_relative_path": os.path.relpath(source_path, os.getcwd()),
                    "run_label": run_label,
                    "filename": filename,
                    "kind": "python" if filename.endswith(".py") else "config",
                    "sha1": digest,
                    "source": source,
                }
            )
    return placeholders


def _resolve_relative_module(
    current_module: str,
    level: int,
    module: str | None,
    module_index: dict[str, str],
) -> str:
    if level == 0:
        return (module or "").strip(".")
    parts = current_module.split(".")
    current_path = module_index.get(current_module, "")
    is_package = current_path.endswith("__init__.py")
    if is_package:
        base = parts[: max(len(parts) - level + 1, 0)]
    else:
        base = parts[:-level]
    target = ".".join(base + ([module] if module else []))
    return target.strip(".")


def _top_level_target_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names.append(node.target.id)
    return names


def _module_docstring(tree: ast.AST) -> str:
    doc = ast.get_docstring(tree) or ""
    return doc.strip().split("\n", 1)[0][:240]


def _parse_module_info(path: str, module_name: str, source: str, tree: ast.AST) -> dict[str, Any]:
    imports: list[dict[str, Any]] = []
    exported_imports: dict[str, dict[str, Any]] = {}
    symbol_defs: dict[str, dict[str, Any]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    {
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname,
                        "lineno": getattr(node, "lineno", 1),
                        "scope": "local" if not isinstance(getattr(node, "parent", None), ast.Module) else "top_level",
                    }
                )
        elif isinstance(node, ast.ImportFrom):
            imports.append(
                {
                    "type": "from",
                    "module": node.module,
                    "level": node.level,
                    "names": [{"name": alias.name, "alias": alias.asname} for alias in node.names],
                    "lineno": getattr(node, "lineno", 1),
                    "scope": "local" if not isinstance(getattr(node, "parent", None), ast.Module) else "top_level",
                }
            )
            if isinstance(getattr(node, "parent", None), ast.Module):
                for alias in node.names:
                    exported_imports[alias.asname or alias.name] = {
                        "module": node.module,
                        "level": node.level,
                        "name": alias.name,
                    }

    for node in tree.body:
        for name in _top_level_target_names(node):
            symbol_defs[name] = {
                "kind": type(node).__name__,
                "lineno": getattr(node, "lineno", 1),
            }

    return {
        "module": module_name,
        "path": path,
        "docstring": _module_docstring(tree),
        "imports": imports,
        "exported_imports": exported_imports,
        "symbol_defs": symbol_defs,
        "source": source,
    }


def _annotate_parents(tree: ast.AST) -> None:
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent


def _load_module_info(path: str, module_name: str, source_override: str | None = None) -> dict[str, Any]:
    source = source_override if source_override is not None else _read_text(path)
    tree = ast.parse(source, filename=path)
    _annotate_parents(tree)
    return _parse_module_info(path, module_name, source, tree)


def _read_dependency_source(path: str, dependency_graph: dict[str, Any]) -> str:
    placeholder_sources = dependency_graph.get("placeholder_sources", {})
    if path in placeholder_sources:
        return placeholder_sources[path]
    return _read_text(path)


def _resolve_reexport(
    module_name: str,
    symbol_name: str,
    module_index: dict[str, str],
    parsed: dict[str, dict[str, Any]],
    seen: set[tuple[str, str]],
    source_overrides: dict[str, str] | None = None,
) -> tuple[str | None, str]:
    if (module_name, symbol_name) in seen:
        return None, "cycle"
    seen.add((module_name, symbol_name))

    path = module_index.get(module_name)
    if not path:
        return None, "missing_module"
    source_overrides = source_overrides or {}
    info = parsed.setdefault(module_name, _load_module_info(path, module_name, source_overrides.get(path)))

    if f"{module_name}.{symbol_name}" in module_index:
        return f"{module_name}.{symbol_name}", "submodule"
    if symbol_name in info["symbol_defs"]:
        return module_name, "symbol"
    exported = info["exported_imports"].get(symbol_name)
    if not exported:
        return None, "unresolved_symbol"

    base_module = _resolve_relative_module(
        module_name,
        exported.get("level", 0),
        exported.get("module"),
        module_index,
    )
    if not base_module:
        return None, "empty_reexport"
    if f"{base_module}.{exported['name']}" in module_index:
        return f"{base_module}.{exported['name']}", "submodule"
    if base_module in module_index:
        resolved, kind = _resolve_reexport(
            base_module,
            exported["name"],
            module_index,
            parsed,
            seen,
            source_overrides,
        )
        return resolved or base_module, f"reexport:{kind}"
    return None, "external_reexport"


def trace_python_dependencies(
    entry_path: str,
    scope_root: str,
    supplemental_placeholders: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    scope_root = os.path.abspath(scope_root)
    entry_path = os.path.abspath(entry_path)
    module_index = _build_module_index(scope_root)
    supplemental_placeholders = supplemental_placeholders or []
    placeholder_sources = {
        item["path"]: item["source"]
        for item in supplemental_placeholders
        if item.get("kind") == "python" and item.get("source")
    }
    placeholder_source_paths = {
        item["path"]: item["source_path"]
        for item in supplemental_placeholders
        if item.get("kind") == "python" and item.get("source_path")
    }
    for path in placeholder_sources:
        module_index[_virtual_module_name(path, scope_root)] = path
    entry_module = _module_name_for_path(entry_path, scope_root)
    parsed: dict[str, dict[str, Any]] = {}
    visited: set[str] = set()
    queue = [entry_module]
    edges: list[dict[str, Any]] = []
    unresolved_imports: list[dict[str, Any]] = []
    external_dependencies: set[str] = set()
    symbol_refs: list[dict[str, Any]] = []

    while queue:
        module_name = queue.pop(0)
        if module_name in visited or module_name not in module_index:
            continue
        visited.add(module_name)
        info = parsed.setdefault(
            module_name,
            _load_module_info(
                module_index[module_name],
                module_name,
                placeholder_sources.get(module_index[module_name]),
            ),
        )

        for imp in info["imports"]:
            if imp["type"] == "import":
                target = imp["module"]
                if target in module_index:
                    edges.append(
                        {
                            "from": module_name,
                            "to": target,
                            "kind": "module",
                            "lineno": imp["lineno"],
                            "scope": imp["scope"],
                        }
                    )
                    queue.append(target)
                else:
                    external_dependencies.add(target.split(".", 1)[0])
            else:
                base_module = _resolve_relative_module(
                    module_name,
                    imp.get("level", 0),
                    imp.get("module"),
                    module_index,
                )
                if base_module and base_module in module_index:
                    edges.append(
                        {
                            "from": module_name,
                            "to": base_module,
                            "kind": "from_import_base",
                            "lineno": imp["lineno"],
                            "scope": imp["scope"],
                        }
                    )
                    queue.append(base_module)

                for imported in imp["names"]:
                    resolved_module, resolution_kind = _resolve_reexport(
                        base_module,
                        imported["name"],
                        module_index,
                        parsed,
                        set(),
                        placeholder_sources,
                    )
                    ref = {
                        "source_module": module_name,
                        "imported_from": base_module,
                        "name": imported["name"],
                        "alias": imported["alias"],
                        "lineno": imp["lineno"],
                        "scope": imp["scope"],
                        "resolved_module": resolved_module,
                        "resolution_kind": resolution_kind,
                    }
                    symbol_refs.append(ref)
                    if resolved_module and resolved_module in module_index:
                        edges.append(
                            {
                                "from": module_name,
                                "to": resolved_module,
                                "kind": resolution_kind,
                                "lineno": imp["lineno"],
                                "scope": imp["scope"],
                                "symbol": imported["name"],
                            }
                        )
                        queue.append(resolved_module)
                    elif base_module:
                        if base_module in module_index:
                            unresolved_imports.append(ref)
                        else:
                            external_dependencies.add(base_module.split(".", 1)[0])

    for path in placeholder_sources:
        module_name = _virtual_module_name(path, scope_root)
        if module_name in visited or module_name not in module_index:
            continue
        try:
            parsed[module_name] = _load_module_info(path, module_name, placeholder_sources[path])
        except SyntaxError:
            parsed[module_name] = {
                "module": module_name,
                "path": path,
                "docstring": "",
                "imports": [],
                "exported_imports": {},
                "symbol_defs": {},
                "source": placeholder_sources[path],
            }
        visited.add(module_name)

    modules = [
        {
            "module": module,
            "path": module_index[module],
            "docstring": parsed[module]["docstring"],
            "symbol_defs": parsed[module]["symbol_defs"],
            "is_placeholder": module_index[module] in placeholder_sources,
            "source_path": placeholder_source_paths.get(module_index[module], ""),
        }
        for module in sorted(visited)
    ]

    return {
        "entry_module": entry_module,
        "entry_path": entry_path,
        "scope_root": scope_root,
        "modules": modules,
        "module_index": module_index,
        "edges": edges,
        "external_dependencies": sorted(external_dependencies),
        "unresolved_imports": unresolved_imports,
        "symbol_refs": symbol_refs,
        "placeholder_sources": placeholder_sources,
        "supplemental_placeholders": [
            {
                key: value
                for key, value in item.items()
                if key != "source"
            }
            for item in supplemental_placeholders
        ],
    }


def parse_report_markdown(report_path: str) -> dict[str, Any]:
    text = _read_text(report_path)
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Code Report"

    hypotheses: list[dict[str, Any]] = []
    pattern = re.compile(
        r"^##\s+(Hypothesis\s+\d+)\n(.*?)(?=^##\s+Hypothesis\s+\d+|\n---\n|\n#\s+Tests and Results)",
        flags=re.MULTILINE | re.DOTALL,
    )
    for idx, match in enumerate(pattern.finditer(text), start=1):
        block = match.group(2).strip()
        statement = ""
        statement_match = re.search(r"\*\*Hypothesis:\*\*\s*(.+)", block)
        if statement_match:
            statement = statement_match.group(1).strip()
        hypotheses.append(
            {
                "id": f"hypothesis_{idx}",
                "title": match.group(1).strip(),
                "statement": statement,
                "body": block,
            }
        )

    tests_match = re.search(
        r"^#\s+Tests and Results\s*\n(.*?)(?=^#\s+Agent Behavior Analysis)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    tests_and_results = tests_match.group(1).strip() if tests_match else ""

    behavior_match = re.search(
        r"^#\s+Agent Behavior Analysis\s*\n(.*)$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    behavior_text = behavior_match.group(1).strip() if behavior_match else ""

    consensus_sections: dict[str, str] = {}
    section_specs = [
        ("what", "What is the agent doing?"),
        ("how", "How is it doing it?"),
        ("why", "Why does it exhibit this behavior?"),
    ]
    for idx, (key, heading) in enumerate(section_specs):
        next_heading = section_specs[idx + 1][1] if idx + 1 < len(section_specs) else None
        if next_heading:
            match = re.search(
                rf"^##\s+{re.escape(heading)}\s*\n(.*?)(?=^##\s+{re.escape(next_heading)}|\Z)",
                behavior_text,
                flags=re.MULTILINE | re.DOTALL,
            )
        else:
            match = re.search(
                rf"^##\s+{re.escape(heading)}\s*\n(.*)$",
                behavior_text,
                flags=re.MULTILINE | re.DOTALL,
            )
        consensus_sections[key] = match.group(1).strip() if match else ""

    claims: list[dict[str, Any]] = []
    for item in hypotheses:
        claims.append(
            {
                "id": item["id"],
                "kind": "hypothesis",
                "title": item["title"],
                "text": item["statement"] or item["body"],
            }
        )
    if tests_and_results:
        claims.append(
            {
                "id": "tests_and_results",
                "kind": "tests_results",
                "title": "Tests and Results",
                "text": tests_and_results,
            }
        )
    for key, value in consensus_sections.items():
        claims.append(
            {
                "id": f"consensus_{key}",
                "kind": "consensus",
                "title": key,
                "text": value,
            }
        )

    return {
        "title": title,
        "hypotheses": hypotheses,
        "tests_and_results": tests_and_results,
        "behavior_analysis": behavior_text,
        "consensus_sections": consensus_sections,
        "claims": claims,
        "raw_text": text,
    }


def _normalize_report_data(report_data: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "title": "Integrated Report",
        "hypotheses": [],
        "tests_and_results": "",
        "behavior_analysis": "",
        "consensus_sections": {},
        "claims": [],
        "raw_text": "",
    }
    if report_data:
        base.update(report_data)
    if base.get("hypotheses") is None:
        base["hypotheses"] = []
    if base.get("consensus_sections") is None:
        base["consensus_sections"] = {}
    if base.get("claims") is None:
        base["claims"] = []
    return base


def build_code_context_bundle_from_report_data(
    *,
    benchmark_entry: str,
    benchmark_config: str,
    code_scope_root: str,
    report_data: dict[str, Any] | None = None,
    report_path: str = "",
    data_path: str = "",
) -> dict[str, Any]:
    normalized_report = _normalize_report_data(report_data)
    supplemental_placeholders = discover_supplemental_code_placeholders(
        data_path=data_path,
        code_scope_root=code_scope_root,
    )
    dependency_graph = trace_python_dependencies(
        benchmark_entry,
        code_scope_root,
        supplemental_placeholders=supplemental_placeholders,
    )
    config_traces = extract_config_traces(benchmark_config, dependency_graph)
    return {
        "report_path": os.path.abspath(report_path) if report_path else "",
        "benchmark_entry": os.path.abspath(benchmark_entry),
        "benchmark_config": os.path.abspath(benchmark_config),
        "code_scope_root": os.path.abspath(code_scope_root),
        "report_data": normalized_report,
        "dependency_graph": dependency_graph,
        "config_traces": config_traces,
        "supplemental_code_placeholders": dependency_graph["supplemental_placeholders"],
        "unresolved_items": dependency_graph["unresolved_imports"],
    }


def _flatten_yaml_keys(data: Any, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            keys.append(full_key)
            keys.extend(_flatten_yaml_keys(value, full_key))
    return keys


def _search_config_hits(source: str, key: str) -> list[str]:
    patterns = [
        rf"\[\s*[\"']{re.escape(key)}[\"']\s*\]",
        rf"get\(\s*[\"']{re.escape(key)}[\"']",
        rf"setdefault\(\s*[\"']{re.escape(key)}[\"']",
        rf"{re.escape(key)}",
    ]
    hits: list[str] = []
    for line_no, line in enumerate(source.splitlines(), start=1):
        if any(re.search(pattern, line) for pattern in patterns):
            hits.append(f"L{line_no}: {line.strip()}")
    return hits[:8]


def extract_config_traces(config_path: str, dependency_graph: dict[str, Any]) -> dict[str, Any]:
    config = yaml.safe_load(_read_text(config_path)) or {}
    keys = sorted(set(_flatten_yaml_keys(config)))
    base_keys = {key.split(".", 1)[0] for key in keys}
    all_keys = sorted(set(keys) | base_keys)

    traces: dict[str, list[dict[str, Any]]] = {}
    modules_by_name = {
        module["module"]: module["path"]
        for module in dependency_graph.get("modules", [])
    }
    for key in all_keys:
        key_hits: list[dict[str, Any]] = []
        leaf = key.split(".")[-1]
        for module_name, path in modules_by_name.items():
            source = _read_dependency_source(path, dependency_graph)
            exact_hits = _search_config_hits(source, key)
            leaf_hits = [] if leaf == key else _search_config_hits(source, leaf)
            chosen = exact_hits or leaf_hits
            if chosen:
                key_hits.append(
                    {
                        "module": module_name,
                        "path": path,
                        "matches": chosen,
                    }
                )
        traces[key] = key_hits

    return {
        "config_path": os.path.abspath(config_path),
        "keys": all_keys,
        "traces": traces,
    }

def build_code_context_bundle(
    report_path: str,
    benchmark_entry: str,
    benchmark_config: str,
    code_scope_root: str,
    data_path: str = "",
) -> dict[str, Any]:
    report_data = parse_report_markdown(report_path)
    return build_code_context_bundle_from_report_data(
        report_path=report_path,
        benchmark_entry=benchmark_entry,
        benchmark_config=benchmark_config,
        code_scope_root=code_scope_root,
        report_data=report_data,
        data_path=data_path,
    )


def _summarize_dependency_graph(bundle: dict[str, Any]) -> dict[str, Any]:
    graph = bundle["dependency_graph"]
    return {
        "entry_module": graph["entry_module"],
        "module_count": len(graph["modules"]),
        "edge_count": len(graph["edges"]),
        "external_dependencies": graph["external_dependencies"],
        "unresolved_imports": len(graph["unresolved_imports"]),
        "key_modules": [module["module"] for module in graph["modules"][:12]],
    }


def _config_flow_summary(bundle: dict[str, Any], config_keys: list[str]) -> dict[str, Any]:
    traces = bundle["config_traces"]["traces"]
    summary: dict[str, Any] = {}
    for key in config_keys:
        summary[key] = [
            {
                "module": hit["module"],
                "path": hit["path"],
                "matches": hit["matches"][:3],
            }
            for hit in traces.get(key, [])
        ][:4]
    return summary


def _risk_control_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    findings = []
    for module in bundle["dependency_graph"]["modules"]:
        source = _read_dependency_source(module["path"], bundle["dependency_graph"]).lower()
        matched = [pattern for pattern in RISK_CONTROL_PATTERNS if pattern in source]
        if matched:
            findings.append(
                {
                    "module": module["module"],
                    "path": module["path"],
                    "matched_terms": matched,
                }
            )
    return {
        "matched_modules": findings,
        "absence_of_controls": not findings,
    }


def _metrics_path_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    interesting = ("sharpe", "aggregate_results", "MetricsEvaluator", "PortfolioAnalyzer")
    hits = []
    for module in bundle["dependency_graph"]["modules"]:
        source = _read_dependency_source(module["path"], bundle["dependency_graph"])
        matched = [term for term in interesting if term in source]
        if matched:
            hits.append(
                {
                    "module": module["module"],
                    "path": module["path"],
                    "matched_terms": matched,
                }
            )
    return {"metrics_modules": hits[:10]}


def build_code_context_digest(bundle: dict[str, Any]) -> str:
    graph = bundle["dependency_graph"]
    focus_keys = ("allow_short", "net_arch", "reward_scaling", "relative_return_alpha", "total_timesteps")
    focus_config = {
        key: [
            {
                "module": hit["module"],
                "matches": hit["matches"][:2],
            }
            for hit in bundle["config_traces"]["traces"].get(key, [])[:2]
        ]
        for key in focus_keys
    }
    risk = _risk_control_summary(bundle)
    metrics = _metrics_path_summary(bundle)
    key_modules = [module["module"] for module in graph["modules"][:8]]
    risk_modules = [item["module"] for item in risk["matched_modules"][:6]]
    metric_modules = [item["module"] for item in metrics["metrics_modules"][:6]]
    return (
        "=== CODE CONTEXT DIGEST ===\n"
        f"Entry module: {graph['entry_module']}\n"
        f"Reachable modules: {len(graph['modules'])}\n"
        f"Key modules: {', '.join(key_modules) or '(none)'}\n"
        f"Metric-path hints: {', '.join(metric_modules) or '(none)'}\n"
        f"Risk-control hints: {', '.join(risk_modules) or '(none found)'}\n"
        "Focused config traces:\n"
        f"{json.dumps(focus_config, indent=2)}\n"
        "=== END CODE CONTEXT DIGEST ==="
    )


def summarize_enriched_claims(enriched_claims: list[dict[str, Any]], limit: int = 8) -> str:
    if not enriched_claims:
        return "(none)"
    items: list[str] = []
    for idx, claim in enumerate(enriched_claims[:limit], start=1):
        items.append(
            "\n".join(
                [
                    f"Claim {idx}: {claim.get('title', '(untitled)')} [{claim.get('kind', 'unknown')}]",
                    f"Text: {claim.get('claim_text', '')}",
                    f"Code paths: {', '.join(claim.get('code_paths', [])) or '(none)'}",
                    f"Config keys: {', '.join(claim.get('config_keys', [])) or '(none)'}",
                    f"Flow: {claim.get('exercised_flow', '(not specified)')}",
                    f"Why: {claim.get('explanation', '(not specified)')}",
                ]
            )
        )
    return "\n\n".join(items)


def summarize_code_evidence_results(evidence_results: list[dict[str, Any]], limit: int = 5) -> str:
    if not evidence_results:
        return "(none)"
    items: list[str] = []
    for idx, item in enumerate(evidence_results[:limit], start=1):
        result = item.get("result", {})
        snippets = result.get("evidence_snippets", [])
        items.append(
            "\n".join(
                [
                    f"Evidence {idx}: {item.get('title', '(untitled)')}",
                    f"Summary: {result.get('summary', '(not available)')}",
                    f"Supporting paths: {', '.join(result.get('supporting_paths', [])) or '(none)'}",
                    f"Config keys: {', '.join(result.get('config_keys', [])) or '(none)'}",
                    f"Snippets: {' | '.join(snippets) if snippets else '(none)'}",
                    f"Confidence: {result.get('confidence', '(unknown)')}",
                ]
            )
        )
    return "\n\n".join(items)


def run_read_only_evidence_tasks(
    tasks: list[dict[str, Any]],
    report_path: str,
    benchmark_entry: str,
    benchmark_config: str,
    code_scope_root: str,
) -> dict[str, Any]:
    bundle = build_code_context_bundle(
        report_path=report_path,
        benchmark_entry=benchmark_entry,
        benchmark_config=benchmark_config,
        code_scope_root=code_scope_root,
    )
    results = []
    for task in tasks:
        task_type = task["task_type"]
        if task_type == "dependency_summary":
            payload = _summarize_dependency_graph(bundle)
        elif task_type == "config_flow":
            payload = _config_flow_summary(bundle, task.get("config_keys", []))
        elif task_type == "risk_controls":
            payload = _risk_control_summary(bundle)
        elif task_type == "metrics_path":
            payload = _metrics_path_summary(bundle)
        else:
            payload = {"error": f"Unknown task type: {task_type}"}
        results.append(
            {
                "task_id": task["id"],
                "title": task["title"],
                "task_type": task_type,
                "result": payload,
            }
        )
    return {"results": results}
