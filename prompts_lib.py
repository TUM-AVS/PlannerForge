"""Build extraction prompts from section-marker templates.

Each template under `extraction_prompt_files/<name>_prompt.txt` is divided
into sections delimited by lines like `[TASK]`, `[CP]`, and `[ICL]`. This
module reads the template, selects the sections requested by the caller,
and returns the joined prompt text.

Usage:

    from prompts_lib import build_prompt

    # The CP+ICL version (equivalent to the original prompt as shipped).
    txt = build_prompt("location", techniques=["cp", "icl"])

    # Bare zero-shot baseline — just the task description.
    txt = build_prompt("location", techniques=[])

    # Tag prompt has a {tag_list} placeholder filled from a sibling file.
    txt = build_prompt("tag", techniques=["cp", "icl"])

The `techniques` list controls which optional sections appear. TASK is
always included. Sections that aren't present in the template are
silently skipped, so a baseline template with no [ICL] still works.
"""
from __future__ import annotations

import re
from pathlib import Path

_HERE = Path(__file__).parent
PROMPTS_DIR = _HERE / "extraction_prompt_files"

# Required + optional section IDs the assembler knows about. Order
# matters — TASK first, then CP elaborates, then ICL shows examples,
# then COT issues the "think step-by-step before answering" instruction
# (placed last so the model sees it just before producing its output,
# the typical CoT-prompt position).
_TASK = "TASK"
_OPTIONAL = ("CP", "ICL", "COT")

# Maps the techniques argument values to section IDs. Lowercase to make
# the API forgiving.
_TECHNIQUE_TO_SECTION = {
    "cp":  "CP",
    "icl": "ICL",
    "cot": "COT",
}


def _section_pattern() -> re.Pattern:
    """One-shot regex compile for `[SECTION_ID]` headers on their own
    line."""
    return re.compile(r"^\[([A-Z][A-Z0-9_]*)\]\s*$", re.MULTILINE)


def _parse_sections(text: str) -> dict[str, str]:
    """Split `text` into a dict of section_id -> body. Sections without
    a header are dropped; if the first line is not a header the leading
    text is also dropped."""
    pat = _section_pattern()
    matches = list(pat.finditer(text))
    if not matches:
        # No markers at all — treat the whole file as TASK so legacy
        # untemplated prompts still work.
        return {_TASK: text.strip()}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_id = m.group(1)
        body = text[start:end].strip("\n")
        sections[section_id] = body
    return sections


def _load_template(name: str) -> str:
    """Read `extraction_prompt_files/<name>_prompt.txt` from disk."""
    path = PROMPTS_DIR / f"{name}_prompt.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _resolve_placeholders(text: str, context: dict[str, str] | None) -> str:
    """Substitute {placeholder} occurrences using `context`. Unknown
    placeholders are left as literal text (no f-string surprises)."""
    if not context:
        return text
    # `tag_list` is the only known placeholder today, but the
    # implementation is general.
    for key, value in context.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_prompt(
    name: str,
    techniques: list[str] | tuple[str, ...] | None = None,
    *,
    context: dict[str, str] | None = None,
    template_text: str | None = None,
) -> str:
    """Assemble the prompt for `name` with the requested techniques.

    Args:
        name: prompt template basename, e.g. "location" → reads
            extraction_prompt_files/location_prompt.txt.
        techniques: list of technique keys to include. Recognised values:
            "cp", "icl". Case-insensitive. Order doesn't matter — output
            sections always follow TASK → CP → ICL ordering.
        context: optional substitution dict for {placeholder} tokens
            inside the template (used by the tag prompt for {tag_list}).
        template_text: bypass the disk read entirely — useful for tests
            that want to feed a template string directly.

    Returns:
        The assembled prompt string. Sections are joined by blank lines
        and the trailing newline is trimmed.

    Raises:
        FileNotFoundError: if the template doesn't exist and no
            template_text override was given.
        ValueError: if a requested technique key is not recognised.
    """
    if template_text is None:
        template_text = _load_template(name)

    # Auto-inject tag_list for the tag prompt when no explicit context
    # is given — keeps callers from having to know about the placeholder.
    context = dict(context or {})
    if name == "tag" and "tag_list" not in context:
        tl_path = PROMPTS_DIR / "tag_list.txt"
        if tl_path.exists():
            context["tag_list"] = tl_path.read_text(encoding="utf-8")

    template_text = _resolve_placeholders(template_text, context)
    sections = _parse_sections(template_text)

    techniques = techniques or []
    selected_section_ids: list[str] = []
    for t in techniques:
        key = t.strip().lower()
        if key not in _TECHNIQUE_TO_SECTION:
            raise ValueError(
                f"unknown technique {t!r}; expected one of "
                f"{sorted(_TECHNIQUE_TO_SECTION)}")
        selected_section_ids.append(_TECHNIQUE_TO_SECTION[key])

    # Always include TASK first; then append optional sections in canonical
    # order regardless of caller's list order. Skip optional sections that
    # weren't selected OR that the template doesn't define.
    out_parts: list[str] = []
    if _TASK in sections:
        out_parts.append(sections[_TASK])
    for section_id in _OPTIONAL:
        if section_id in selected_section_ids and section_id in sections:
            out_parts.append(sections[section_id])

    return "\n\n".join(p.strip() for p in out_parts if p.strip())


# ── Convenience aliases ────────────────────────────────────────────────

# The 6-condition ablation matrix. Conditions are functions of (prompt
# techniques, decoding strategy); this dict only captures the prompt-
# content half. SC is a runtime decoding choice handled by the runner.
CONDITIONS = {
    "baseline":         [],                       # task only — zero-shot
    "cp":               ["cp"],                   # + domain context / rules
    "cp_cot":           ["cp", "cot"],            # + chain-of-thought
    "cp_icl":           ["cp", "icl"],            # + in-context examples
    "cp_icl_cot":       ["cp", "icl", "cot"],     # + all prompt-text techniques
    "cp_icl_cot_sc":    ["cp", "icl", "cot"],     # same prompt; SC by sampler
}
