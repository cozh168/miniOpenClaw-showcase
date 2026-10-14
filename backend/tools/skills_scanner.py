from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import yaml
from service.workbench_files import MAX_TEXT_BYTES, write_text_file


logger = logging.getLogger(__name__)


@dataclass
class SkillRecord:
    name: str
    description: str
    path: str


def _parse_frontmatter(text: str) -> dict[str, str]:
    lines = text.removeprefix("\ufeff").splitlines()
    if not lines or lines[0].rstrip(" \t") != "---":
        return {}
    closing = next((index for index in range(1, len(lines)) if lines[index].rstrip(" \t") == "---"), None)
    if closing is None:
        raise ValueError("SKILL.md frontmatter is missing its closing --- separator")
    try:
        data = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise ValueError("SKILL.md frontmatter must contain valid YAML") from exc
    if not isinstance(data, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    return {str(key): str(value) for key, value in data.items()}


def validate_skill_frontmatter(text: str) -> None:
    _parse_frontmatter(text)


def scan_skills(base_dir: Path) -> list[SkillRecord]:
    skills_dir = base_dir / "skills"
    records: list[SkillRecord] = []
    if not skills_dir.exists() or skills_dir.is_symlink():
        return records

    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        if skill_file.is_symlink() or skill_file.parent.is_symlink() or skill_file.parent.name.startswith("."):
            continue
        try:
            if skill_file.stat().st_size > MAX_TEXT_BYTES:
                logger.warning("Skipping oversized skill: %s", skill_file.parent.name)
                continue
            text = skill_file.read_text(encoding="utf-8")
            meta = _parse_frontmatter(text)
        except (OSError, UnicodeError, ValueError):
            logger.warning("Skipping unreadable or invalid skill: %s", skill_file.name if skill_file.parent == skills_dir else skill_file.parent.name)
            continue
        records.append(
            SkillRecord(
                name=meta.get("name", skill_file.parent.name),
                description=meta.get("description", "No description"),
                path=str(skill_file.relative_to(base_dir)).replace("\\", "/"),
            )
        )
    return records


def build_snapshot(skills: list[SkillRecord]) -> str:
    lines = [
        "<skills>",
        "  <summary>Available local skills that the agent can inspect with read_file.</summary>",
    ]
    for skill in skills:
        lines.extend(
            [
                f"  <skill name={quoteattr(skill.name)} path={quoteattr(skill.path)}>",
                f"    <description>{escape(skill.description)}</description>",
                "  </skill>",
            ]
        )
    lines.append("</skills>")
    return "\n".join(lines) + "\n"


def refresh_snapshot(base_dir: Path) -> Path:
    skills_dir = base_dir / "skills"
    if skills_dir.is_symlink():
        raise ValueError("Symbolic skill directories are not supported")
    skills_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = skills_dir / "SKILLS_SNAPSHOT.md"
    if snapshot_path.is_symlink():
        raise ValueError("Symbolic skill snapshots are not supported")
    write_text_file(snapshot_path, build_snapshot(scan_skills(base_dir)))
    return snapshot_path
