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
# Must match the [max_tokens] entries in conf/config.toml. This is the budget for any
# task whose name is NOT a key in that table, which is most of them -- 23 of the 24
# eval tasks. When the two disagree, the budget a task gets depends on whether
# someone happened to name it in the config, and for a reasoning model the budget
# decides the score: `filename` (a config key) and `filename_leak` (not one) send the
# same 185-character prompt, and nemotron scored 0% and 100% on them respectively.
DEFAULT_MAX_TOKENS = 32000
DEFAULT_TIMEOUT = 600

# Model families for quirks
MODEL_FAMILIES = ["qwopus", "qwen", "gemma", "nemotron", "laguna", "foundation"]

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
