# LLM API Constants

# Default host/port
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1337

# Default model
DEFAULT_MODEL = "foundation"

# API endpoints
API_TAGS = "/api/tags"
API_GENERATE = "/api/generate"
API_CHAT = "/api/chat"

# Request defaults
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TIMEOUT = 600

# Model families for quirks
MODEL_FAMILIES = ["qwen", "gemma", "nemotron", "laguna", "foundation"]

# Timeouts per task type
TIMEOUTS = {
    "think": 600,
    "json": 600,
    "summarize": 300,
    "filename": 120,
    "vlm": 600,
}

# Max tokens per task
MAX_TOKENS = {
    "think": 16000,
    "json": 16000,
    "summarize": 16000,
    "filename": 1000,
    "vlm": 16000,
}