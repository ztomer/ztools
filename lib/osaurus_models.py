#!/usr/bin/env python3
import os
import requests
from typing import List, Optional



DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1337


def get_api_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}/v1/chat/completions"


def get_base_url(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def get_models(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = "") -> List[str]:
    try:
        if host.startswith("http"):
            url = f"{host}/v1/models"
        else:
            url = f"http://{host}:{port}/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.get(url, timeout=10, headers=headers)
        if resp.status_code == 200:
            return [m["id"] for m in resp.json().get("data", [])]
    except requests.exceptions.Timeout:
        pass
    except requests.exceptions.ConnectionError:
        pass
    except Exception as e:
        print(f"Warning: get_models failed: {e}")
    return []


def is_server_running(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> bool:
    try:
        if host.startswith("http"):
            url = f"{host}/v1/models"
        else:
            url = f"http://{host}:{port}/v1/models"
        resp = requests.get(url, timeout=3)
        return resp.status_code in (200, 404)
    except requests.exceptions.Timeout:
        return False
    except requests.exceptions.ConnectionError:
        return False
    except Exception as e:
        print(f"Warning: server check failed: {e}")
        return False


def check_llm_availability(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, api_key: str = "") -> bool:
    return is_server_running(host, port)


get_available_models = get_models


def get_best_model(task: str = None, env_var: str = "OLLAMA_MODEL") -> str:
    from .config import get_best_model as _get_best
    if task:
        return os.environ.get(env_var, _get_best(task))
    return os.environ.get(env_var, "foundation")


def select_best_vlm_model(available_models: List[str]) -> Optional[str]:
    vlm_keywords = ["vl", "vision", "qwen", "llamavl"]
    for keyword in vlm_keywords:
        for model in available_models:
            if keyword.lower() in model.lower():
                return model
    return None


def select_best_model(models: list, preferred: list = None) -> str:
    if not models:
        return None
    if preferred is None:
        preferred = ["foundation", "qwen", "gemma"]
    for pref in preferred:
        for model in models:
            if pref.lower() in model.lower():
                return model
    return models[0] if models else None
