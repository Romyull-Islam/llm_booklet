# week4_agent/tools.py
import inspect, json, math

TOOLS = {}

def tool(fn):
    """Register a python function as a tool, building its JSON schema from type hints."""
    sig = inspect.signature(fn)
    type_map = {int: "integer", float: "number", str: "string", bool: "boolean"}
    props, required = {}, []
    for name, p in sig.parameters.items():
        props[name] = {"type": type_map.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            required.append(name)
    TOOLS[fn.__name__] = {
        "fn": fn,
        "schema": {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        },
    }
    return fn

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '3 * (2 + 1)'. Use this for any arithmetic."""
    allowed = {k: v for k, v in vars(math).items() if not k.startswith("_")}
    return str(eval(expression, {"__builtins__": {}}, allowed))

@tool
def word_count(text: str) -> str:
    """Count the words in a piece of text."""
    return str(len(text.split()))
