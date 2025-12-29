from abc import ABC

from loguru import logger

from ..context import C
from ..flow import BaseFlow
from ..schema import ServiceConfig


class BaseService(ABC):

    def __init__(self, service_config: ServiceConfig):
        self.service_config: ServiceConfig = service_config

    def integrate_flow(self, _flow: BaseFlow) -> bool:
        return False

    def integrate_stream_flow(self, _flow: BaseFlow) -> bool:
        return False

    def run(self):
        flow_names = []
        for _, flow in C.flow_dict.items():
            if flow.stream:
                if self.integrate_stream_flow(flow):
                    flow_names.append(flow.name)

            else:
                if self.integrate_flow(flow):
                    flow_names.append(flow.name)

        logger.info(f"integrate {','.join(flow_names)}")

        import warnings

        warnings.filterwarnings("ignore", category=DeprecationWarning)
