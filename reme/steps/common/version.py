"""Return the package version."""

from ..base_step import BaseStep

from ...components import R


@R.register("version_step")
class VersionStep(BaseStep):
    """Emit reme.__version__ as the response answer."""

    async def execute(self):
        assert self.context is not None
        from ... import __version__

        self.logger.info(f"[{self.name}] version={__version__}")
        self.context.response.answer = __version__
        self.context.response.metadata["version"] = __version__
        return self.context.response
