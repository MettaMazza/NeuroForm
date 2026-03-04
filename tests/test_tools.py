import pytest
from unittest.mock import MagicMock, patch
from neuroform.brain.orchestrator import BrainOrchestrator
from neuroform.tools.manager import tool_registry

# Reset registry for clean test
@pytest.fixture(autouse=True)
def clean_registry():
    tool_registry._tools = {}
    tool_registry._schemas = []
    
    # Register a dummy tool
    def dummy_math(x: int, y: int) -> str:
        return str(x + y)
        
    tool_registry.register(
        func=dummy_math,
        description="Adds two numbers.",
        parameters={"x": {"type": "integer"}, "y": {"type": "integer"}},
        requires_owner=False
    )
    yield

def test_tool_registry_schemas():
    schemas = tool_registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "dummy_math"

def test_tool_registry_execute():
    result = tool_registry.execute("dummy_math", {"x": 5, "y": 7})
    assert result == "12"
    
    # Non-existent tool
    err_result = tool_registry.execute("fake_tool", {})
    assert "not found" in err_result

@patch("ollama.chat")
def test_orchestrator_tool_loop(mock_ollama_chat):
    # Mock orchestrator
    kg_mock = MagicMock()
    orch = BrainOrchestrator(kg=kg_mock)
    
    # Simulate first ollama response calling the tool via generic text JSON
    mock_ollama_chat.side_effect = [
        {
            "model": "gemma3:4b",
            "message": {
                "role": "assistant",
                "content": 'I need to use a tool for this.\n[TOOL: dummy_math(x=10, y=20)]'
            }
        },
        # Simulate second ollama response delivering the final answer based on tool result
        {
            "model": "gemma3:4b",
            "message": {
                "role": "assistant",
                "content": "The result of 10 plus 20 is 30."
            }
        },
        # Observer-Critic audit response (ALLOWED)
        {
            "model": "gemma3:4b",
            "message": {
                "role": "assistant",
                "content": '{"verdict": "ALLOWED", "reason": "Safe", "guidance": "None"}'
            }
        }
    ]
    
    # Execute loop
    final_response = orch._execute_inference_with_tools(
        user_id="U123",
        message="What is 10 plus 20?",
        scope="PRIVATE",
        tiered_ctx="[CONTEXT]"
    )
    
    assert final_response == "The result of 10 plus 20 is 30."
    assert mock_ollama_chat.call_count == 3  # 2 inference + 1 observer-critic audit
    
    # Verify the conversation history fed to the second call
    second_call_args = mock_ollama_chat.call_args_list[1][1]
    messages = second_call_args["messages"]
    
    assert messages[0]["role"] == "system"
    assert "You have access to the following tools" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "USER MESSAGE:\nWhat is 10 plus 20?" in messages[1]["content"]
    assert messages[2]["role"] == "assistant"
    assert "TOOL: dummy_math" in messages[2]["content"]
    assert messages[3]["role"] == "user"
    assert "TOOL OBSERVATION FOR dummy_math:\n30" in messages[3]["content"]
