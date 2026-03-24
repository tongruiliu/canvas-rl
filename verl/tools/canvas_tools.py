"""Canvas-task tool schemas for RL-side registration."""

from typing import Any


canvas_tools: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "insert_element",
            "description": (
                "Insert a new SVG or HTML element into the notebook blackboard. "
                "Use this for initial construction or adding new objects."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fragment": {
                        "type": "string",
                        "description": (
                            "The HTML or SVG string to insert. "
                            "If using SVG, it must contain xmlns='http://www.w3.org/2000/svg'. "
                            "The inserted tag must include a unique id."
                        ),
                    },
                    "rootId": {
                        "type": "string",
                        "description": "Parent container id. Use 'root' unless a different parent is required.",
                    },
                    "beforeId": {
                        "type": "string",
                        "description": "Optional sibling id to insert before.",
                    },
                },
                "required": ["fragment", "rootId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_element",
            "description": "Update attributes or text content of an existing element without redrawing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targetId": {
                        "type": "string",
                        "description": "The id of the element to modify.",
                    },
                    "attrs": {
                        "type": "object",
                        "description": (
                            "Key-value attributes to update. "
                            "Use key='text' to replace the inner text of a text-like element."
                        ),
                    },
                },
                "required": ["targetId", "attrs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_element",
            "description": "Remove one element from the notebook blackboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targetId": {
                        "type": "string",
                        "description": "The id of the element to remove.",
                    }
                },
                "required": ["targetId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_element",
            "description": "Replace one existing element with a new HTML or SVG fragment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targetId": {
                        "type": "string",
                        "description": "The id of the element being replaced.",
                    },
                    "fragment": {
                        "type": "string",
                        "description": "The new HTML or SVG fragment.",
                    },
                },
                "required": ["targetId", "fragment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear",
            "description": "Clear the whole notebook blackboard. Use only for full reset.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]