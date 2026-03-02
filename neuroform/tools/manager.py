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

    def register(self, func: Callable, description: str, parameters: Dict[str, Dict[str, str]]):
        """
        Registers a Python function as an LLM-accessible tool.
        
        Args:
            func: The Python function to call
            description: What the tool does
            parameters: Dict mapping arg_name -> {"type": "string|integer", "description": "..."}
        """
        name = func.__name__
        self._tools[name] = func
        
        # Build schema in standard OpenAI/Ollama format
        schema = {
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
        logger.info(f"Registered tool: {name}")

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Returns the list of tool schemas to inject into the LLM request."""
        return self._schemas

    def get_prompt_instructions(self) -> str:
        """Returns a string description of all tools for the system prompt."""
        if not self._schemas:
            return ""
            
        instructions = "You have access to the following tools:\n\n"
        for schema in self._schemas:
            func = schema["function"]
            instructions += f"- {func['name']}: {func['description']}\n"
            instructions += "  Arguments:\n"
            for arg, meta in func["parameters"]["properties"].items():
                instructions += f"    - {arg} ({meta['type']}): {meta.get('description', '')}\n"
            instructions += "\n"
            
        instructions += (
            "To use a tool, you MUST output a JSON block containing EXACTLY one tool call, like this:\n"
            "```json\n"
            "{\n"
            '  "tool_call": {\n'
            '    "name": "<tool_name>",\n'
            '    "arguments": {\n'
            '      "<arg_name>": "<arg_value>"\n'
            '    }\n'
            '  }\n'
            "}\n"
            "```\n"
            "Wait for the tool result observation before continuing your response. Do NOT output a tool call if you do not need one. If no tool is needed, just answer normally."
        )
        return instructions

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        """
        Executes a registered tool and returns the result as a string.
        """
        if name not in self._tools:
            error_msg = f"Tool '{name}' not found."
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
