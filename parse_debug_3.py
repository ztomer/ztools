with open("llm_raw_dump.txt", "r") as f:
    text = f.read()

# find all sections
sections = text.split("=== RES ===")
for i, sec in enumerate(sections):
    if not sec.strip(): continue
    raw = sec.split("=== END RES ===")[0]
    with open(f"section_{i}.txt", "w") as f:
        f.write(raw)
    print(f"Wrote section {i}")
