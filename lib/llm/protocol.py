from typing import Protocol, Optional, Any


class LLMClient(Protocol):
    """Protocol defining the standard LLM call interface.

    All LLM backends (osaurus, llm.client, mlx_lib) must conform.
    """

    def call(
        self,
        model: str,
        messages: list[dict],
        *,
        host: str = "localhost",
        port: int = 1337,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        task: str = "think",
        parse_json: bool = False,
    ) -> dict: ...


def create_client(client_type: str = "osaurus") -> LLMClient:
    """Factory: return a callable conforming to the LLMClient protocol."""
    if client_type == "osaurus":
        from lib.osaurus_lib import call
        return call
    elif client_type == "mlx":
        from lib.mlx_lib import call
        return call
    elif client_type == "llm":
        from lib.llm.client import call
        return call
    raise ValueError(f"Unknown client type: {client_type}")
