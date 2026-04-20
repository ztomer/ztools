from lib.osaurus_lib import _extract_json_only
with open('test_mlx_output.txt', 'w') as f:
    f.write("""<think>
Here's a thinking process:
...
   ```json
   {
     "fixed_activities": [
       {
         "name": "Ontario Science Centre",
         "location": "Toronto",
         "target_ages": "6-13",
         "price": "$18-35 per child or free",
         "weather": "indoor"
       }
     ]
   }
   ```
...
   `{ "fixed_activities": [ { "name": "Ontario Science Centre", "location": "Toronto", "target_ages": "6-13", "price": "$18-35 per child or free", "weather": "indoor" }, ... ] }`
   Perfect. 
   Output matches response. 
   [Done] 
   *Output Generation* (matches the final JSON)
   ```json
   {
     "fixed_activities": [
       {
         "name": "Ontario Science Centre",
         "location": "Toronto",
         "target_ages": "6-13",
         "price": "$18-35 per child or free",
         "weather": "indoor"
       },
       {
         "name": "Ripley's Aquarium of Canada",
         "
""")

with open('test_mlx_output.txt', 'r') as f:
    text = f.read()

res = _extract_json_only(text)
print("EXTRACTED:")
print(res)

