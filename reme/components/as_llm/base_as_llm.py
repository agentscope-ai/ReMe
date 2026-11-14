"""Shared lifecycle for lazy AgentScope chat providers."""

from agentscope.credential import CredentialBase
from agentscope.model import ChatModelBase

from ..base_component import BaseComponent
from ...enumeration import ComponentEnum


class BaseAsLLM(BaseComponent):
    """Base wrapper for AgentScope chat models.

    Subclasses set ``credential_cls``. Providers are constructed on first use,
    allowing local file and search jobs to run without model credentials.
    """

    component_type = ComponentEnum.AS_LLM
    credential_cls: type[CredentialBase]
    _protect_owned_model_lifecycle = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model: ChatModelBase | None = None
        self._owns_model = False
        self._closed = False
        self._closing_depth = 0

    async def start(self) -> None:
        """Start without cleaning up resources owned by another lifecycle call."""
        await super().start()
        if not self._closing_depth:
            self._closed = False

    async def _rollback_start(
        self,
        start_hook_entered: bool,
        started_owned: list[BaseComponent],
    ) -> None:
        """Fence lazy model access during every startup rollback path."""
        # BaseComponent.start() holds the lifecycle lock, but the synchronous
        # model getter does not. Both dependency and hook failures can yield
        # while releasing an owned client, so temporarily forbid construction.
        self._closing_depth += 1
        self._closed = True
        try:
            await super()._rollback_start(start_hook_entered, started_owned)
            if not start_hook_entered and self._owns_model and self._model is not None:
                try:
                    await self._close()
                except BaseException:
                    self.logger.exception("Failed to clean up model after startup failure")
        finally:
            self._closing_depth -= 1
            if not self._closing_depth:
                # A failed start is retryable, unless another close is pending.
                self._closed = False

    async def close(self) -> None:
        """Prevent lazy construction throughout started or pre-start shutdown."""
        self._closing_depth += 1
        self._closed = True
        try:
            async with self._lock:
                if not self._is_started:
                    await self._close()
                    return
            await super().close()
        finally:
            self._closed = True
            self._closing_depth -= 1

    async def _start(self) -> None:
        """Keep service startup independent of provider credentials."""

    @property
    def model(self) -> ChatModelBase:
        """Initialize on first access, including direct Step dependency resolution."""
        self.initialize_model()
        assert self._model is not None
        return self._model

    @model.setter
    def model(self, value: ChatModelBase | None) -> None:
        """Preserve explicit model injection used by standalone consumers."""
        self._model = value
        self._owns_model = False

    def initialize_model(self) -> None:
        """Construct the configured provider once, without making a remote request."""
        if self._protect_owned_model_lifecycle and self._closed and (self._owns_model or self._model is None):
            raise RuntimeError("Start the LLM component before accessing its model after close.")
        if self._model is not None:
            return
        kwargs = dict(self.kwargs)
        credential = self.credential_cls(**kwargs.pop("credential", {}))
        model_cls = credential.get_chat_model_class()
        params_dict = kwargs.pop("parameters", None)
        parameters = model_cls.Parameters(**params_dict) if params_dict else None
        self._model = model_cls(credential=credential, parameters=parameters, **kwargs)
        self._owns_model = True
