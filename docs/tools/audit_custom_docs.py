#!/usr/bin/env python3
"""
Static compatibility audit for custom/copied docs under ./docs (excluding ./docs/source).

Goal (Route A):
- Keep only docs that are runnable/accurate against the current repository baseline (0.4.3) in docs/ root.
- Archive non-compatible legacy docs under docs/legacy/.

This script performs a best-effort *static* audit:
- Missing internal `python -m ...` modules (lerobot.* / vis.*)
- Missing internal import paths (lerobot.* / vis.*)
- Missing referenced repo paths under src/lerobot/...
- Unknown `lerobot-xxx` CLI commands (based on pyproject.toml [project.scripts])
- Unknown `--robot.type=...` / `--teleop.type=...` values (based on register_subclass in src/)
- Missing referenced runnable scripts like `python path/to/script.py`

It does NOT execute workflows (no hardware, no training run).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import re
import sys


RE_PYTHON_M = re.compile(r"\bpython(?:3)?\s+-m\s+([A-Za-z0-9_.]+)")
RE_PYTHON_SCRIPT = re.compile(r"\bpython(?:3)?\s+([./A-Za-z0-9_-]+\.py)\b")
# Best-effort CLI detector:
# - Avoid false positives like `docs/source/lerobot-dataset-v3.mdx` (filenames) by requiring a sane terminator.
# - Keep this conservative: missing a mention is better than flagging lots of noise.
RE_LEROBOT_CLI = re.compile(r"\blerobot-[a-z0-9][a-z0-9_-]*[a-z0-9](?=(?:\\s|$|[`;,:\\)\\]]))")
RE_IMPORT = re.compile(
    r"(?m)^\s*(?:from\s+([A-Za-z0-9_.]+)\s+import|import\s+([A-Za-z0-9_.]+))"
)
RE_SRC_PATH = re.compile(r"(src/lerobot/[A-Za-z0-9_./-]+)")
RE_ROBOT_TYPE = re.compile(r"--robot\.type(?:=|\s+)([A-Za-z0-9_]+)")
RE_TELEOP_TYPE = re.compile(r"--teleop\.type(?:=|\s+)([A-Za-z0-9_]+)")

# Targeted legacy-only camera markers.
# Keep this conservative: we only want to flag docs that instruct to *use* the missing backend.
RE_ROS_CAMERA_MARKERS = re.compile(
    r"(?:['\"]type['\"]\s*:\s*['\"]ros_camera['\"])|(?:\bRosCameraConfig\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    kind: str
    detail: str


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return p
    raise RuntimeError(f"Could not find repo root from {start}")


def parse_project_scripts(pyproject_text: str) -> set[str]:
    scripts: set[str] = set()
    in_section = False
    for raw in pyproject_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line == "[project.scripts]"
            continue
        if not in_section:
            continue
        # Example: lerobot-record = "lerobot.scripts.lerobot_record:main"
        m = re.match(r"([A-Za-z0-9_-]+)\s*=", line)
        if m:
            scripts.add(m.group(1))
    return scripts


def collect_choice_registry_types(src_root: Path, decorator_regex: re.Pattern[str]) -> set[str]:
    types: set[str] = set()
    for path in src_root.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in decorator_regex.finditer(text):
            types.add(m.group(1))
    return types


def module_exists(repo_root: Path, module: str) -> bool:
    # Only check internal modules for this repo.
    if module.startswith("lerobot."):
        rel = Path("src") / Path(*module.split("."))
        return (repo_root / f"{rel}.py").is_file() or (repo_root / rel / "__init__.py").is_file()
    if module.startswith("vis."):
        rel = Path(*module.split("."))
        return (repo_root / f"{rel}.py").is_file() or (repo_root / rel / "__init__.py").is_file()
    return True


def is_placeholder_type(type_value: str) -> bool:
    # Docs templates often use placeholders like my_robot/your_robot.
    if type_value in {"my_robot", "your_robot", "robot", "teleop"}:
        return True
    if type_value.startswith("my_") or type_value.startswith("your_"):
        return True
    return False


def is_placeholder_module(module: str) -> bool:
    parts = module.split(".")
    return any(p.startswith(("my_", "your_")) for p in parts)


def is_placeholder_path(raw_path: str) -> bool:
    parts = Path(raw_path).parts
    if any(p.startswith(("my_", "your_")) for p in parts):
        return True
    # Common doc placeholders.
    if "..." in parts:
        return True
    if any("<" in p or ">" in p for p in parts):
        return True
    return False


def script_path_exists(repo_root: Path, doc_dir: Path, raw_path: str) -> bool:
    p = Path(raw_path)
    if (repo_root / p).is_file():
        return True
    if (doc_dir / p).is_file():
        return True
    return False


def audit_file(
    repo_root: Path,
    doc_path: Path,
    known_cli: set[str],
    robot_types: set[str],
    teleop_types: set[str],
) -> list[Finding]:
    findings: list[Finding] = []

    try:
        text = doc_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [Finding(kind="read_error", detail=str(e))]

    # Skip self-referential / meta docs that intentionally mention missing legacy artifacts.
    ignore_legacy_markers = doc_path.as_posix().endswith("docs/legacy/README.md")

    # `python -m ...`
    for mod in RE_PYTHON_M.findall(text):
        if is_placeholder_module(mod):
            continue
        if mod.startswith(("lerobot.", "vis.")) and not module_exists(repo_root, mod):
            findings.append(Finding(kind="missing_python_module", detail=mod))

    # `import ...` / `from ... import ...`
    for m in RE_IMPORT.finditer(text):
        mod = m.group(1) or m.group(2) or ""
        if is_placeholder_module(mod):
            continue
        if mod.startswith(("lerobot.", "vis.")) and not module_exists(repo_root, mod):
            findings.append(Finding(kind="missing_import", detail=mod))

    # `src/lerobot/...` paths
    for raw in RE_SRC_PATH.findall(text):
        if is_placeholder_path(raw):
            continue
        p = repo_root / raw
        if not p.exists():
            findings.append(Finding(kind="missing_src_path", detail=raw))

    # `lerobot-xxx` CLI
    for cli in RE_LEROBOT_CLI.findall(text):
        if cli not in known_cli:
            findings.append(Finding(kind="unknown_cli", detail=cli))

    # `--robot.type=...`
    for t in RE_ROBOT_TYPE.findall(text):
        if is_placeholder_type(t):
            continue
        if t not in robot_types:
            findings.append(Finding(kind="unknown_robot_type", detail=t))

    # `--teleop.type=...`
    for t in RE_TELEOP_TYPE.findall(text):
        if is_placeholder_type(t):
            continue
        if t not in teleop_types:
            findings.append(Finding(kind="unknown_teleop_type", detail=t))

    # `python path/to/script.py`
    for raw in RE_PYTHON_SCRIPT.findall(text):
        # Avoid double-reporting `python -m ...` (already handled) and inline `python ./-m` etc.
        if raw.startswith("-"):
            continue
        if is_placeholder_path(raw):
            continue
        if not script_path_exists(repo_root, doc_path.parent, raw):
            findings.append(Finding(kind="missing_script", detail=raw))

    # Targeted ROS camera markers
    if not ignore_legacy_markers and RE_ROS_CAMERA_MARKERS.search(text):
        # Baseline repo does not include ros_camera backend.
        findings.append(Finding(kind="unknown_camera_type", detail="ros_camera"))

    # Deduplicate for stable reporting
    uniq: dict[tuple[str, str], Finding] = {}
    for f in findings:
        uniq[(f.kind, f.detail)] = f
    return sorted(uniq.values(), key=lambda x: (x.kind, x.detail))


def summarize_findings(findings: list[Finding], max_items: int = 3) -> str:
    if not findings:
        return "—"
    parts: list[str] = []
    for f in findings[:max_items]:
        parts.append(f"{f.kind}: {f.detail}")
    more = len(findings) - len(parts)
    if more > 0:
        parts.append(f"(+{more})")
    return "; ".join(parts)


def main() -> int:
    repo_root = find_repo_root(Path(__file__).resolve())
    docs_root = repo_root / "docs"

    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8", errors="replace")
    known_cli = parse_project_scripts(pyproject_text)

    robot_types = collect_choice_registry_types(
        repo_root / "src" / "lerobot" / "robots",
        re.compile(r"@RobotConfig\.register_subclass\(\"([^\"]+)\"\)"),
    )
    teleop_types = collect_choice_registry_types(
        repo_root / "src" / "lerobot" / "teleoperators",
        re.compile(r"@TeleoperatorConfig\.register_subclass\(\"([^\"]+)\"\)"),
    )

    excluded_files = {
        (docs_root / "DOCS_AUDIT_REPORT.md").resolve(),
        (docs_root / "CUSTOM_DOCS_INDEX_0.4.3.md").resolve(),
    }

    scanned: list[Path] = []
    for p in docs_root.rglob("*"):
        if not p.is_file():
            continue
        if p.resolve() in excluded_files:
            continue
        # Exclude official doc-builder sources and internal tooling under docs/,
        # even when they are symlinks (Path.resolve() would escape `docs/source`).
        try:
            rel = p.relative_to(docs_root)
        except ValueError:
            # Shouldn't happen (p comes from docs_root.rglob), but keep defensive.
            continue
        if rel.parts and rel.parts[0] in {"source", "tools"}:
            continue
        # Skip obvious binary artifacts.
        if p.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif", ".mp4"}:
            continue
        scanned.append(p)
    scanned.sort()

    results: list[tuple[Path, str, list[Finding]]] = []
    counts = Counter()
    top_missing = Counter()

    for path in scanned:
        findings = audit_file(repo_root, path, known_cli, robot_types, teleop_types)

        rel_to_docs = path.relative_to(docs_root).as_posix()
        is_legacy = rel_to_docs.startswith("legacy/")
        is_draft = "/temp_docs/" in rel_to_docs

        if is_legacy:
            # Route A: anything under docs/legacy is treated as archived (not runnable on baseline).
            if findings:
                status = "Legacy (archived, incompatible)"
                counts["Legacy incompatible"] += 1
                for f in findings:
                    top_missing[f"{f.kind}: {f.detail}"] += 1
            else:
                status = "Legacy (archived, draft)" if is_draft else "Legacy (archived)"
                counts["Legacy archived"] += 1
        else:
            if findings:
                status = "Not compatible"
                counts["Not compatible"] += 1
                for f in findings:
                    top_missing[f"{f.kind}: {f.detail}"] += 1
            else:
                status = "OK (draft)" if is_draft else "OK"
                counts["OK"] += 1
        results.append((path, status, findings))

    # Emit Markdown report
    rel = lambda p: p.relative_to(repo_root).as_posix()
    lines: list[str] = []
    lines.append("# 自定义文档兼容性审计报告 (lerobot 0.4.3)")
    lines.append("")
    lines.append(f"- 审计日期: {date.today().isoformat()}")
    lines.append("- 审计对象: `docs/` 下除 `docs/source/` 之外的文档/脚本/说明文件")
    # Version is in pyproject.toml; keep it simple here.
    lines.append("- 基线代码: 当前仓库 `pyproject.toml` 显示版本 `0.4.3`")
    lines.append("")
    lines.append("## 结论概览")
    lines.append("")
    lines.append(f"- 扫描文件数: {len(scanned)}")
    lines.append(f"- docs/ 根目录可直接使用 (OK): {counts['OK']}")
    lines.append(f"- docs/ 根目录不匹配 (Not compatible): {counts['Not compatible']}")
    lines.append(f"- docs/legacy/ 归档文件数: {counts['Legacy archived'] + counts['Legacy incompatible']}")
    lines.append(f"  - 其中静态发现不匹配点: {counts['Legacy incompatible']}")
    lines.append("")
    lines.append("> 说明: 这是静态审计(模块/路径/CLI 入口存在性 + 明显缺失检查)，没有运行硬件/训练/回放流程。")
    lines.append("")
    lines.append("## 逐文件初判 (按路径)")
    lines.append("")
    lines.append("| 文件 | 状态 | 静态不匹配点(截断) |")
    lines.append("|---|---|---|")
    for path, status, findings in results:
        lines.append(f"| `{rel(path)}` | {status} | {summarize_findings(findings)} |")

    if top_missing:
        lines.append("")
        lines.append("## 缺失引用 Top 列表 (用于集中修复)")
        lines.append("")
        for item, n in top_missing.most_common(40):
            lines.append(f"- `{item}` (出现 {n} 次)")

    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
