import sys
from lib.mlx_lib import find_text_mlx_model, call_mlx
mlx_model = find_text_mlx_model(["qwen"])
if mlx_model:
    print(f"Model: {mlx_model.name}")
    sys_prompt = "Output JSON now. Use EXACT schema: {\"fixed_activities\": [{\"name\": \"str\", \"location\": \"str\", \"target_ages\": \"str\", \"price\": \"str\", \"weather\": \"str\"}]}"
    usr_prompt = "Extract popular Vaughan/Toronto venues for families with kids ages 6-13. MANDATORY default values: target_ages: \"6-13\", price: $18-35 per child or free, weather: \"indoor\". Never leave any field empty."
    raw = call_mlx(mlx_model, f"System: {sys_prompt}\n\nUser: {usr_prompt}")
    print("RAW MLX CONTENT:")
    print("---")
    print(raw)
    print("---")
else:
    print("No MLX model found")
