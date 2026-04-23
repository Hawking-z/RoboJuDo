from __future__ import annotations

from abc import ABC


class PerceptionProvider(ABC):
    def attach_to_spec(self, spec) -> None:
        return None

    def bind(self, model, data, viewer=None) -> None:
        return None

    def refresh(self) -> dict[str, object]:
        return {}

    def render_debug(self, viewer=None) -> None:
        return None

    def close(self) -> None:
        return None
