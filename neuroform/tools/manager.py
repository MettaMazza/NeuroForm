"""
Tool Manager — Registration and Schema Generation
=================================================
Manages all native Python tools that Nero can call autonomously.
Automatically generates JSON schemas compatible with Ollama/Gemini tool calling.
"""
import inspect
import json
import logging
from typing import Callable, Dict, Any, List

logger = logging.getLogger(__name__)

class ToolManager:
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._schemas: List[Dict[str, Any]] = []
        self._tool_ownership: Dict[str, bool] = {}

    def register(self, func: Callable, description: str, parameters: Dict[str, Dict[str, str]], requires_owner: bool = True):
        """
        Registers a Python function as an LLM-accessible tool.
        
        Args:
            func: The Python function to call
            description: What the tool does
            parameters: Dict mapping arg_name -> {"type": "string|integer", "description": "..."}
            requires_owner: If True, only the system owner can invoke it. Defaults to True for safety.
        """
        name = func.__name__
        self._tools[name] = func
        self._tool_ownership[name] = requires_owner
        
        # Build schema in standard OpenAI/Ollama format
        schema: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": list(parameters.keys())
                }
            }
        }
        
        for arg, meta in parameters.items():
            schema["function"]["parameters"]["properties"][arg] = {
                "type": meta.get("type", "string"),
                "description": meta.get("description", "")
            }
            
        self._schemas.append(schema)
        logger.info(f"Registered tool: {name} (Requires owner: {requires_owner})")

    def get_schemas(self, is_owner: bool = False) -> List[Dict[str, Any]]:
        """Returns the list of tool schemas, filtered by user permissions."""
        if is_owner:
            return self._schemas
        return [s for s in self._schemas if not self._tool_ownership.get(s["function"]["name"], True)]

    def get_prompt_instructions(self, is_owner: bool = False) -> str:
        """Returns a string description of all tools for the system prompt."""
        schemas = self.get_schemas(is_owner)
        if not schemas:
            return ""
            
        instructions = "You have access to the following tools:\n\n"
        for schema in schemas:
            func = schema["function"]
            instructions += f"- {func['name']}: {func['description']}\n"
            instructions += "  Arguments:\n"
            for arg, meta in func["parameters"]["properties"].items():
                instructions += f"    - {arg} ({meta['type']}): {meta.get('description', '')}\n"
            instructions += "\n"
            
        instructions += (
            "To use a tool, you MUST use the following exact syntax:\n"
            '[TOOL: <tool_name>(<arg_name>="<arg_value>")]\n'
            "For example:\n"
            '[TOOL: duckduckgo_search(query="latest news")]\n'
            "Wait for the tool result observation before continuing your response. Do NOT output a tool call if you do not need one. If no tool is needed, just answer normally."
        )
        return instructions

    def execute(self, name: str, arguments: Dict[str, Any], is_owner: bool = False) -> str:
        """
        Executes a registered tool and returns the result as a string.
        """
        if name not in self._tools:
            error_msg = f"Tool '{name}' not found."
            logger.warning(error_msg)
            return error_msg
            
        # Hard block if user tries to hallucinate/force an owner tool without privileges
        if not is_owner and self._tool_ownership.get(name, True):
            error_msg = f"Error: Tool '{name}' requires OWNER privileges to execute."
            logger.warning(error_msg)
            return error_msg
            
        func = self._tools[name]
        try:
            logger.info(f"Executing tool '{name}' with args: {arguments}")
            result = func(**arguments)
            return str(result)
        except Exception as e:
            error_msg = f"Error executing '{name}': {str(e)}"
            logger.error(error_msg)
            return error_msg

# Global instance
tool_registry = ToolManager()
