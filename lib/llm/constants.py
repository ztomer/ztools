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
MODEL_FAMILIES = ["qwopus", "qwen", "gemma", "nemotron", "laguna", "foundation"]

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

# Client type constants
DEFAULT_CLIENT_TYPE = "osaurus"
CLIENT_TYPE_OSAURUS = "osaurus"
CLIENT_TYPE_MLX = "mlx"
CLIENT_TYPE_LLM = "llm"

# Standard tasks, roles, and status codes
TASK_THINK = "think"
ROLE_USER = "user"
ROLE_SYSTEM = "system"
HTTP_STATUS_OK = 200

# Quirks-specific prompt and model constants
DEFAULT_FAMILY = "default"
QWEN_FAMILY = "qwen"
GEMMA4_FAMILY = "gemma4"
QWEN_TRIGGER_PREFIX = "Output JSON now"
QWEN_TRIGGER_TEXT = "Output JSON now.\n\n"
NO_JSON_KW = "no json"
PLAIN_TEXT_KW = "plain text"
GEMMA4_PREFIX_KW = "JSON"
GEMMA4_PREFIX_IMPORTANT = "IMPORTANT"
GEMMA4_TRIGGER_TEXT = "IMPORTANT: This is DATA EXTRACTION. Output JSON only. "
USER_REPLACE_EXECUTE = "execute"
USER_REPLACE_CONTEXT = "context"
REPLACE_SRC_CONTEXT = "Current Context"
REPLACE_TGT_CONTEXT = "Data"
REPLACE_SRC_TASK = "Execute the task"
REPLACE_TGT_TASK = "Extract to JSON"
REPLACE_SRC_TASK_BASED = "Execute the task based on"
REPLACE_TGT_TASK_BASED = "Extract"
