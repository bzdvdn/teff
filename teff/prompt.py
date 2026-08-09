"""Prompt template rendering from workflow state."""

from typing import Any


class _TemplateDict(dict):
    def __missing__(self, key: str) -> Any:
        msg = f"prompt template references unknown state key: {key!r}"
        raise KeyError(msg)


def render_template(template: str, state: dict) -> str:
    """Render ``{key}`` placeholders in *template* from *state*.

    Values are coerced to strings, so ``{summ}`` with an ``int`` value
    renders as-is.  A placeholder referencing a missing state key raises
    ``KeyError`` so template mistakes surface early.

    Usage::

        render_template(
            "create a repair plan for {type} up to {summ}",
            {"type": "kitchen", "summ": 15000},
        )
        # "create a repair plan for kitchen up to 15000"
    """
    if "{" not in template:
        return template
    values = {k: str(v) for k, v in state.items()}
    return template.format_map(_TemplateDict(values))
