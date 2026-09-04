from typing import Optional, Protocol

from lib.llm.constants import (
    CLIENT_TYPE_LLM,
    CLIENT_TYPE_MLX,
    CLIENT_TYPE_OSAURUS,
    DEFAULT_CLIENT_TYPE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TEMPERATURE,
    TASK_THINK,
)


class LLMClient(Protocol):
    """Protocol defining the standard LLM call interface.

    All LLM backends (osaurus, llm.client, mlx_lib) must conform.
    """

    def call(
        self,
        model: str,
        messages: list[dict],
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        task: str = TASK_THINK,
        parse_json: bool = False,
    ) -> dict: ...


class OsaurusClient:
    def call(
        self,
        model: str,
        messages: list[dict],
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        task: str = TASK_THINK,
        parse_json: bool = False,
    ) -> dict:
        from lib.osaurus_lib import call as _call

        return _call(
            model,
            messages,
            host=host,
            port=port,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            task=task,
            parse_json=parse_json,
        )


class MlxClient:
    def call(
        self,
        model: str,
        messages: list[dict],
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        task: str = TASK_THINK,
        parse_json: bool = False,
    ) -> dict:
        from lib.mlx_lib import call as _call

        return _call(
            model,
            messages,
            host=host,
            port=port,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            task=task,
            parse_json=parse_json,
        )


class GenericClient:
    def call(
        self,
        model: str,
        messages: list[dict],
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        task: str = TASK_THINK,
        parse_json: bool = False,
    ) -> dict:
        from lib.llm.client import call as _call

        return _call(
            model,
            messages,
            host=host,
            port=port,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            task=task,
            parse_json=parse_json,
        )


def create_client(client_type: str = DEFAULT_CLIENT_TYPE) -> LLMClient:
    """Factory: return an LLMClient instance conforming to the LLMClient protocol."""
    if client_type == CLIENT_TYPE_OSAURUS:
        return OsaurusClient()
    elif client_type == CLIENT_TYPE_MLX:
        return MlxClient()
    elif client_type == CLIENT_TYPE_LLM:
        return GenericClient()
    raise ValueError(f"Unknown client type: {client_type}")
