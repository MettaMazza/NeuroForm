"""
Tool Argument Parser — State-machine parser for tool call arguments.

Parses key=value argument strings from LLM tool calls without using eval().
Handles quoted strings, escape sequences, triple-quotes, and literal types.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

def parse_tool_args(args_str: str) -> dict:
    """
    Parse tool call argument string into a dict.

    Handles: key="value with 'apostrophes' and newlines", mode="overwrite"
    No eval() — uses a state-machine parser that tracks quote depth.
    """
    if not args_str or not args_str.strip():
        return {}

    result: dict[str, Any] = {}
    i: int = 0
    s: str = args_str.strip()
    n: int = len(s)

    while i < n:
        # Skip whitespace and commas between key=value pairs
        while i < n and s[i] in ' ,\t\n\r':
            i += 1
        if i >= n:
            break

        # --- Parse key ---
        key_start = i
        while i < n and s[i] not in '= \t':
            i += 1
        key = s[key_start:i].strip()
        if not key:
            break

        # Skip whitespace before '='
        while i < n and s[i] in ' \t':
            i += 1
        if i >= n or s[i] != '=':
            break
        i += 1  # skip '='

        # Skip whitespace after '='
        while i < n and s[i] in ' \t':
            i += 1
        if i >= n:
            result[key] = ""
            break

        # --- Parse value ---
        if s[i] in ('"', "'"):
            quote_char = s[i]
            # Check for triple-quote
            if i + 2 < n and s[i:i+3] == quote_char * 3:
                end_marker = quote_char * 3
                i += 3
                val_start = i
                end_pos = s.find(end_marker, i)
                if end_pos == -1:
                    result[key] = s[val_start:]
                    break
                result[key] = s[val_start:end_pos]
                i = end_pos + 3
            else:
                # Single-quoted: track escapes
                i += 1
                val_parts = []
                while i < n:
                    if s[i] == '\\' and i + 1 < n:
                        nxt = s[i + 1]
                        if nxt == 'n':
                            val_parts.append('\n')
                        elif nxt == 't':
                            val_parts.append('\t')
                        elif nxt == '\\':
                            val_parts.append('\\')
                        elif nxt == quote_char:
                            val_parts.append(quote_char)
                        else:
                            val_parts.append(s[i:i+2])
                        i += 2
                    elif s[i] == quote_char:
                        i += 1
                        break
                    else:
                        val_parts.append(s[i])
                        i += 1
                result[key] = ''.join(val_parts)
        else:
            # Unquoted value — read until comma or end
            val_start = i
            while i < n and s[i] != ',':
                i += 1
            raw = s[val_start:i].strip()
            # Parse literals
            if raw.lower() == 'true':
                result[key] = True
            elif raw.lower() == 'false':
                result[key] = False
            elif raw.lower() == 'none':
                result[key] = None
            else:
                try:
                    result[key] = int(raw)
                except ValueError:
                    try:
                        result[key] = float(raw)
                    except ValueError:
                        result[key] = raw

    if result:
        return result

    # Last resort: treat entire string as content
    return {"content": args_str}
