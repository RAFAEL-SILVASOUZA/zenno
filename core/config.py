from pathlib import Path

from pydantic_settings import BaseSettings

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:1234/v1"
    ollama_api_key: str = "ollama"
    default_model: str = "llama3.2"
    max_reasoning_iterations: int = 3
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Retry / resilience
    max_api_retries: int = 3
    retry_base_delay: float = 1.0       # seconds
    retry_max_delay: float = 10.0       # seconds
    api_request_timeout: float = 120.0  # seconds

    # Smart routing
    # When the rule-based classifier returns a confidence below this threshold,
    # fall back to the LLM classifier. Lower = trust heuristics more.
    classify_heuristic_threshold: float = 0.7

    # Self-consistency
    # Toggle the whole feature. Per-domain sample counts are in core/domains.py.
    self_consistency_enabled: bool = True
    # Hard cap on samples per iteration so a misconfigured domain can't blow up
    # the ollama queue.
    self_consistency_max_samples: int = 5

    # Sandbox verification
    sandbox_enabled: bool = True
    sandbox_timeout: float = 5.0        # seconds per subprocess

    model_config = {"env_file": str(_ENV_FILE)}


settings = Settings()
