from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..op import BaseOp


def parse_flow_expression(flow_content: str) -> BaseOp:
    from ..context import C
    from ..enumeration import RegistryEnum
    from ..op import BaseOp

    flow_content = flow_content.strip()
    if not flow_content:
        raise ValueError("flow content is empty")

    op_registry = C.registry_dict[RegistryEnum.OP]
    env: dict = {name or cls.__name__: cls for name, cls in op_registry.items()}

    lines = [x.strip() for x in flow_content.splitlines() if x.strip()]
    if len(lines) > 1:
        exec_content = "\n".join(lines[:-1])
        exec(exec_content, {"__builtins__": {}}, env)

    last_line_expr = lines[-1]
    result = eval(last_line_expr, {"__builtins__": {}}, env)
    assert isinstance(result, BaseOp), f"Expression '{last_line_expr}' did not evaluate to a BaseOp instance"
    return result
