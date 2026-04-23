from __future__ import annotations

from robojudo.environment.perception.base import PerceptionProvider


class PerceptionManager:
    def __init__(self, providers: list[PerceptionProvider] | None = None):
        self.providers = providers or []

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    def attach_to_spec(self, spec) -> None:
        for provider in self.providers:
            provider.attach_to_spec(spec)

    def bind(self, model, data, viewer=None) -> None:
        for provider in self.providers:
            provider.bind(model, data, viewer=viewer)

    def refresh(self) -> dict[str, object]:
        outputs: dict[str, object] = {}
        for provider in self.providers:
            outputs.update(provider.refresh())
        return outputs

    def render_debug(self, viewer=None) -> None:
        for provider in self.providers:
            provider.render_debug(viewer=viewer)

    def close(self) -> None:
        for provider in self.providers:
            provider.close()
