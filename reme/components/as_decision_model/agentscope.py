"""Optional AgentScope decision model adapter, initialized only when called."""

import asyncio
from collections.abc import Mapping
from importlib import import_module
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from .base_as_decision_model import BaseAsDecisionModel
from ..component_registry import R
from ...utils.async_utils import complete_in_thread

if TYPE_CHECKING:
    from agentscope.classifier import ClassifierModelBase, ClassifierResponse


class _DecisionOptions(BaseModel):
    """Validate transport options without importing optional provider modules."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, min_length=1)
    timeout: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    max_retries: int = Field(default=2, ge=0)
    retry_delay: float = Field(default=0.5, ge=0, allow_inf_nan=False)


@R.register("agentscope")
class AgentScopeAsDecisionModel(BaseAsDecisionModel):
    """Wrap AgentScope's TypeSafe decision model with lazy provider loading.

    ``questions`` accepts AgentScope question objects or JSON dictionaries with
    a ``type`` discriminator (``binary``, ``choice``, or ``score``). Hosts may inject a borrowed model
    through ``model``; otherwise this component constructs and owns its client.
    """

    def __init__(
        self,
        credential: dict | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.credential = dict(credential or {})
        self.options = _DecisionOptions(
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self._model: "ClassifierModelBase | None" = None
        self._owns_model = False
        self._active_calls = 0
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def model(self) -> "ClassifierModelBase | None":
        """Return the current instance without constructing a provider client."""
        return self._model

    @model.setter
    def model(self, value: "ClassifierModelBase | None") -> None:
        """Borrow a host-owned instance, including via Application.update_component.

        Close the component before replacing an internally owned client. An
        active request must finish before any replacement or detachment.
        """
        if value is self._model:
            return
        if value is not None and not callable(value):
            raise TypeError("An injected decision model must be callable.")
        if self._active_calls or self._owns_model:
            raise RuntimeError("Close the decision component before replacing its active or owned model.")
        self._model = value

    @staticmethod
    def _load_decision_model():
        """Load the optional AgentScope adapter on the first decision request."""
        try:
            module = import_module("agentscope.classifier")
            # Check exports as well as module presence for older installations.
            return module.ClassifierQuestion, module.JevClassifierModel
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "Decision model inference requires compatible AgentScope classifier support. "
                "Install it with `pip install 'reme-ai[decision]'`.",
            ) from exc

    async def __call__(self, state: str | dict, questions: Mapping[str, Any]) -> "ClassifierResponse":
        """Validate the request before constructing or calling the provider."""
        if not isinstance(state, (str, dict)) or (isinstance(state, str) and not state.strip()):
            raise ValueError("Decision state must be non-empty text or a JSON object.")
        if isinstance(state, dict):
            state = TypeAdapter(dict[str, JsonValue]).validate_python(state, strict=True)
            json.dumps(state, allow_nan=False)
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("Decision questions must be a non-empty mapping.")
        if any(not isinstance(name, str) or not name.strip() for name in questions):
            raise ValueError("Decision question names must be non-empty strings.")

        question_type, model_cls = self._load_decision_model()
        adapter = TypeAdapter(question_type)
        parsed = {name: adapter.validate_python(question) for name, question in questions.items()}
        await self.start()
        async with self._lock:
            if not self.is_started:
                raise RuntimeError("Decision model component closed before the request could start.")
            if self._model is None:
                credential = dict(self.credential)
                api_key = credential.get("api_key")
                if not isinstance(api_key, str) or not api_key.strip():
                    raise ValueError("Decision model inference requires DECISION_MODEL_API_KEY or credential.api_key.")
                if not credential.get("base_url"):
                    credential["base_url"] = None
                # Reserve ownership before yielding so host injection cannot race
                # initialization. The worker publishes the model to this component
                # before cancellation can release the lifecycle lock.
                self._owns_model = True
                try:
                    await complete_in_thread(self._create_model, model_cls, credential)
                except ImportError as exc:
                    raise ImportError(
                        "Decision model inference requires a compatible typesafe-sdk. "
                        "Install it with `pip install 'reme-ai[decision]'`.",
                    ) from exc
                finally:
                    self._owns_model = self._model is not None
            model = self._model
            self._active_calls += 1
            self._idle.clear()
        try:
            return await model(state=state, questions=parsed)
        finally:
            # No await here: finishing requests must unblock close(), which owns
            # the lifecycle lock while draining active requests.
            self._active_calls -= 1
            if self._active_calls == 0:
                self._idle.set()

    def _create_model(self, model_cls, credential) -> None:
        """Construct and retain the owned client, including synchronous TLS setup."""
        credential_cls = import_module("agentscope.credential").TypeSafeCredential
        self._model = model_cls(
            credential=credential_cls(**credential),
            **self.options.model_dump(exclude_none=True),
        )

    async def _close(self) -> None:
        """Finish draining and cleanup before propagating shutdown cancellation."""
        cleanup = asyncio.create_task(self._release_model())
        cancellation = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                # Keep the lifecycle lock until cleanup finishes, even after
                # repeated cancellation. No cleanup task outlives close().
                cancellation = exc
        cleanup.result()
        if cancellation is not None:
            raise cancellation

    async def _release_model(self) -> None:
        """Release the owned TypeSafe client and allow a later fresh start."""
        await self._idle.wait()
        model, owned = self._model, self._owns_model
        self._model, self._owns_model = None, False
        if model is not None and owned:
            await model.client.aclose()
