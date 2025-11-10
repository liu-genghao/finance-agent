from typing import TypedDict, Sequence, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
import operator


def merge_dicts(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, d2 values overwrite d1."""
    return {**d1, **d2}
"""
merge_dicts({"a": 1, "b": 2}, {"b": 3, "c": 4})
# => {'a': 1, 'b': 3, 'c': 4}
"""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]
    # Potentially add a field for the initial user query if it needs to be passed around
    # user_query: str

"""
state1 = {
    "messages": [msg1],
    "data": {"x": 1},
    "metadata": {"step": 1}
}

state2 = {
    "messages": [msg2],
    "data": {"y": 2},
    "metadata": {"step": 2}
}
合并时框架会自动调用：
messages: [msg1] + [msg2]
data: merge_dicts({"x":1}, {"y":2}) → {"x":1, "y":2}
metadata: merge_dicts({"step":1}, {"step":2}) → {"step":2}
"""