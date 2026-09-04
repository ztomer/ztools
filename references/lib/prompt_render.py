"""The single prompt renderer, shared by production and the evaluator.

Class C1 / C12 in `docs/REPORT_WEAKNESS_CLASSES.md`. Two defects made every other
quality question unanswerable:

- Production rendered externalised prompts with `str.format()`. The templates
  embed literal JSON schemas, so an unescaped `{` made `.format()` raise, and a
  broad `except` downgraded that to a `.replace("{}", ...)` that substituted
  nothing. The model was handed the literal text `{date_range}`.
- The evaluator rendered the same templates with a *different*, silent helper, so
  `ev` was green on templates production could not render at all.

The fix is one function with three properties:

1. **It never calls `str.format()`.** Substitution is targeted, so a literal JSON
   brace in a template is inert rather than a syntax error.
2. **It raises.** An unrendered placeholder is a loud `PromptRenderError`, never a
   prompt that quietly reaches a model with `{placeholder}` still in it.
3. **Both callers use it.** Drift between eval and production is not a convention
   to remember; there is only one code path.

Two template conventions exist in `conf/models/*.toml` and both are supported
*explicitly*, because guessing between them is what the old `except` did:

- positional -- a single `{}` slot, filled from the `positional` argument
- named ------ `{location}`, `{age_range}`, `{date_range}`, filled from **fields
"""

from __future__ import annotations

import re

__all__ = ["PromptRenderError", "render_prompt", "unrendered_placeholders"]

# A placeholder is a bare lower-snake identifier OR a numeric format field in
# braces. A JSON schema brace (`{"transient_events": ...`) does not match, which
# is exactly why substitution is done by targeted replace rather than by
# str.format(). Numeric fields are included because `{0}` reaching a model is the
# same class of failure as `{date_range}` reaching one.
_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*|\d+)\}")

POSITIONAL_SLOT = "{}"


class PromptRenderError(ValueError):
    """A template could not be fully rendered.

    Raised instead of shipping a partially-substituted prompt. Callers that can
    genuinely proceed without the template must catch this and say so in the
    output -- never swallow it into a silent fallback.
    """


def unrendered_placeholders(text: str) -> list[str]:
    """Placeholder names still present in `text`, in first-seen order."""
    seen: dict[str, None] = {}
    for name in _PLACEHOLDER.findall(text):
        seen.setdefault(name, None)
    return list(seen)


def render_prompt(
    template: str,
    *,
    template_id: str = "<inline>",
    positional: str | None = None,
    **fields: object,
) -> str:
    """Render `template`, or raise `PromptRenderError`.

    Args:
        template: the raw template text.
        template_id: e.g. `"qwen.toml:weekend_transient"`. Appears in the error so
            a failure names the file to edit rather than just the symptom.
        positional: value for the legacy single `{}` slot. Required if, and only
            if, the template contains one.
        **fields: named placeholder values.

    Raises:
        PromptRenderError: the template needs a positional value that was not
            supplied, a positional value was supplied for a template with no slot,
            or any placeholder survived substitution.
    """
    if not isinstance(template, str):
        raise PromptRenderError(
            f"{template_id}: template must be str, got {type(template).__name__}"
        )

    has_slot = POSITIONAL_SLOT in template
    if has_slot and positional is None:
        raise PromptRenderError(
            f"{template_id}: template uses the positional '{{}}' slot but no "
            f"positional value was supplied"
        )
    if positional is not None and not has_slot:
        raise PromptRenderError(
            f"{template_id}: a positional value was supplied but the template has "
            f"no '{{}}' slot -- it likely uses named placeholders "
            f"{unrendered_placeholders(template)}"
        )

    result = template
    if has_slot:
        result = result.replace(POSITIONAL_SLOT, str(positional))
    for name, value in fields.items():
        result = result.replace("{" + name + "}", str(value))

    left = unrendered_placeholders(result)
    if left:
        raise PromptRenderError(
            f"{template_id}: unrendered placeholder(s) {left}; "
            f"supplied fields were {sorted(fields)}"
        )
    return result
