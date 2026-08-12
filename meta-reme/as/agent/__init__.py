"""AgentScope agents for diagnosing and optimizing Meta-ReMe candidates.

Import concrete modules directly. Keeping this initializer side-effect free
avoids a cycle because the optimizer owns a tool that invokes the diagnostic
agent.
"""
