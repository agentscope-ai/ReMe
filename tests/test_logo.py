"""test logo"""

from reme.core.schema import ServiceConfig, MCPConfig

if __name__ == "__main__":
    from reme.core.utils import print_logo

    c = ServiceConfig(app_name="reme", backend="mcp", mcp=MCPConfig(transport="sse"))
    print_logo(service_config=c)
