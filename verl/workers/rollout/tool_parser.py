"""Tool-call parsers used to recover structured tool invocations from model outputs."""

import json
import re
from dataclasses import dataclass


@dataclass
class FunctionCall:
    """One parsed tool call emitted by the model."""

    arguments: str
    name: str


class ToolParser:
    _registry: dict[str, type["ToolParser"]] = {}

    def __init__(self, tokenizer) -> None:
        """Bind a tokenizer so parsers can decode model output ids."""
        self.tokenizer = tokenizer

    def extract_tool_calls(self, response_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """Return remaining assistant text plus parsed tool calls."""
        raise NotImplementedError

    @classmethod
    def get_tool_parser(cls, name: str, tokenizer):
        """Construct a registered parser by name."""
        if name not in cls._registry:
            raise ValueError(f"Unknown tool parser: {name}")
        return cls._registry[name](tokenizer)

    @classmethod
    def register(cls, name: str):
        """Decorator for registering concrete parser implementations."""
        def decorator(subclass: type["ToolParser"]) -> type["ToolParser"]:
            cls._registry[name] = subclass
            return subclass

        return decorator


@ToolParser.register("hermes")
class HermesToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        """Parser for `<tool_call> ... </tool_call>` style outputs."""
        super().__init__(tokenizer)
        self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def extract_tool_calls(self, response_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """Decode the response and recover Hermes-format tool invocations."""
        text = self.tokenizer.decode(response_ids)
        matches = self.tool_call_regex.findall(text)
        if not matches:
            return text, []

        function_calls = []
        for match in matches:
            try:
                payload = json.loads(match)
                function_calls.append(
                    FunctionCall(
                        name=payload["name"],
                        arguments=json.dumps(payload["arguments"], ensure_ascii=False),
                    )
                )
            except Exception:
                continue

        content = self.tool_call_regex.sub("", text)
        return content, function_calls


@ToolParser.register("gpt-oss")
class GptOssToolParser(ToolParser):
    def __init__(self, tokenizer) -> None:
        """Parser for gpt-oss harmony-style function call outputs."""
        super().__init__(tokenizer)
        self.cot_pattern = re.compile(
            r"<\\|start\\|>assistant<\\|channel\\|>analysis<\\|message\\|>.*?<\\|end\\|>",
            re.DOTALL,
        )
        self.partial_cot_pattern = re.compile(r"<\\|channel\\|>analysis<\\|message\\|>(.*?)<\\|end\\|>", re.DOTALL)
        self.tool_call_pattern = re.compile(
            r"<\\|start\\|>assistant<\\|channel\\|>[^<]* to=functions\\.([^<]+) "
            r"<\\|constrain\\|>json<\\|message\\|>(.*?)<\\|call\\|>",
            re.DOTALL,
        )

    def extract_tool_calls(self, response_ids: list[int]) -> tuple[str, list[FunctionCall]]:
        """Strip analysis content and recover gpt-oss function calls."""
        text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        if self.tokenizer.pad_token:
            text = text.replace(self.tokenizer.pad_token, "")
        text = re.sub(self.cot_pattern, "", text)
        text = re.sub(self.partial_cot_pattern, "", text)

        matches = re.findall(self.tool_call_pattern, text)
        if not matches:
            return text, []

        function_calls = [FunctionCall(name=name, arguments=arguments) for name, arguments in matches]
        content = re.sub(self.tool_call_pattern, "", text)
        return content, function_calls