from typing import TYPE_CHECKING

from ..enumeration import ComponentEnum
from ..schema import ApplicationConfig
from ..utils import PydanticConfigParser

if TYPE_CHECKING:
    from .base_component import BaseComponent


class ApplicationContext:

    def __init__(self, *args, config: str = "", **kwargs):
        parser = PydanticConfigParser(config_class=ApplicationConfig)
        self.app_config: ApplicationConfig = parser.parse_args(*args, config=config, **kwargs)
        self.components: dict[ComponentEnum, dict[str, BaseComponent]] = {}

        from .component_registry import R

        for component_type, component_configs in self.app_config.components.items():
            self.components[component_type] = {
                name: R.get(component_type, config.get("backend"))(**config)
                for name, config in component_configs.items()
            }
