import importlib
from dataclasses import fields, is_dataclass
from typing import Any

from omegaconf import OmegaConf

from .schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


def _instantiate_dataclass(dataclass_type, payload: dict[str, Any]):
    """Recursively build nested dataclasses from plain dictionaries."""
    if payload is None:
        return dataclass_type()

    kwargs = {}
    for field_info in fields(dataclass_type):
        if field_info.name not in payload:
            continue
        value = payload[field_info.name]
        field_type = field_info.type

        if is_dataclass(field_type) and isinstance(value, dict):
            # recursively build.
            kwargs[field_info.name] = _instantiate_dataclass(field_type, value)
        elif (
            field_info.name == "properties"
            and dataclass_type is OpenAIFunctionParametersSchema
            and isinstance(value, dict)
        ):
            kwargs[field_info.name] = {
                key: _instantiate_dataclass(OpenAIFunctionPropertySchema, val) for key, val in value.items()
            }
        else:
            kwargs[field_info.name] = value
    
    return dataclass_type(**kwargs)

def _build_tool_schema(tool_schema_dict: dict[str, Any]) -> OpenAIFunctionToolSchema:
    """Convert a plain config dict into a normalized tool schema object."""
    function_dict = tool_schema_dict["function"]
    parameters_dict = function_dict.get("parameters", {})
    parameters = _instantiate_dataclass(OpenAIFunctionParametersSchema, parameters_dict)
    function = OpenAIFunctionSchema(
        name=function_dict["name"],
        description=function_dict["description"],
        parameters=parameters,
        strict=function_dict.get("strict", False),
    )
    
    return OpenAIFunctionToolSchema(type=tool_schema_dict["type"], function=function)

def get_tool_class(cls_name: str):
    """Import a tool class from its fully-qualified module path."""
    module_name, class_name = cls_name.rsplit(".", 1)
    module = importlib.import_module(module_name)

    return getattr(module, class_name)

def initialize_tools_from_config(tools_config_file: str) -> list:
    """Load tools from YAML and instantiate them for rollout-time use."""
    tools_config = OmegaConf.load(tools_config_file)
    tool_list = []

    for tool_config in tools_config.tools:
        cls_name = tool_config.class_name
        tool_cls = get_tool_class(cls_name)
        config_dict = OmegaConf.to_container(tool_config.config, resolve=True)
        tool_schema_dict = OmegaConf.to_container(tool_config.get("tool_schema"), resolve=True)
        tool_schema = _build_tool_schema(tool_schema_dict) if tool_schema_dict is not None else None
        tool = tool_cls(config=config_dict, tool_schema=tool_schema)
        tool_list.append(tool)

    return tool_list