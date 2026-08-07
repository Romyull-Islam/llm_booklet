# week4_agent/agent_langchain.py
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOpenAI(
    model="HuggingFaceTB/SmolLM2-135M-Instruct",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    temperature=0.0,
)

def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '3 * (2 + 1)'."""
    import math
    allowed = {k: v for k, v in vars(math).items() if not k.startswith("_")}
    return str(eval(expression, {"__builtins__": {}}, allowed))

agent = create_agent(
    model=model,
    tools=[calculator],
    system_prompt="You are a careful assistant. Use the calculator for all arithmetic.",
    checkpointer=InMemorySaver(),      # gives the agent memory across turns
)

config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 17 * 23 + 4?"}]},
    config,
)
print(result["messages"][-1].content)

# Same thread_id, so it remembers
follow_up = agent.invoke(
    {"messages": [{"role": "user", "content": "Now divide that by 5."}]},
    config,
)
print(follow_up["messages"][-1].content)
