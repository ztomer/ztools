
import os
os.chdir("/Users/ztomer/MLXModels/OsaurusAI")
from mlx_lm import load, stream_generate

model, tokenizer = load("/Users/ztomer/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-mxfp4")

with open("/Users/ztomer/Projects/ztools/.mlx_debug/prompt_30c11dbe.txt", "r") as f:
    prompt = f.read()

# Prepend JSON trigger to avoid thinking
prompt = "Output JSON:\n" + prompt
text_parts = []
for r in stream_generate(model, tokenizer, prompt, max_tokens=2048):
    if hasattr(r, "text"):
        text_parts.append(r.text)
    elif isinstance(r, str):
        text_parts.append(r)
response = "".join(text_parts)
# Strip the trigger from response
if response.startswith("Output JSON:\n"):
    response = response[13:]
print(response, flush=True)
