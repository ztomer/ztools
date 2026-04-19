#!/usr/bin/env python3
"""
Model Quirks Explorer - Systematically probe model behaviors.
Used to discover prompt/response patterns for different models.
"""

import sys
import json
from lib.osaurus_lib import call, get_model_family
from lib.logging_config import osaurus_logger as logger

TEST_PROMPTS = {
    "simple_json": {
        "system": "Output JSON with name and age.",
        "user": "My name is John, I am 25 years old",
    },
    "output_json_now": {
        "system": "Output JSON now. Give name and age.",
        "user": "John is 25",
    },
    "extraction_framing": {
        "system": "IMPORTANT: This is DATA EXTRACTION. Output JSON only.",
        "user": "Extract: John, 25 years old",
    },
    "execute_task": {
        "system": "Execute the task to output JSON with name and age.",
        "user": "John is 25",
    },
    "data_context": {
        "system": "Given the data, output JSON.",
        "user": "Data: John is 25 years old. Extract to JSON.",
    },
    "schema_example": {
        "system": "Output JSON: {\"name\": \"value\", \"age\": number}. Example: John is 25.",
        "user": "Process: John, 25",
    },
    "no_thinking": {
        "system": "Answer immediately in JSON only. No thinking. {\"name\": \"John\", \"age\": 25}",
        "user": "What's 2+2?",
    },
}


def test_model(model: str, test_name: str, timeout: int = 30) -> dict:
    """Run a single test and return result."""
    prompts = TEST_PROMPTS.get(test_name, {})
    if not prompts:
        return {"error": f"Unknown test: {test_name}"}
    
    try:
        result = call(
            model=model,
            messages=[
                {"role": "system", "content": prompts.get("system", "")},
                {"role": "user", "content": prompts.get("user", "")},
            ],
            task="json",
            parse_json=True,  # Request JSON format from server
            timeout=timeout,
        )
        
        # Analyze result
        content = result.get("content", "")
        parsed = result.get("parsed")
        
        analysis = {
            "has_json": bool(parsed),
            "has_json_chars": "{" in content or "[" in content,
            "is_list": isinstance(parsed, list),
            "is_dict": isinstance(parsed, dict),
            "first_keys": list(parsed[0].keys()) if isinstance(parsed, list) and parsed else list(parsed.keys()) if isinstance(parsed, dict) else [],
            "content_first_50": content[:50] if content else "",
            "error": result.get("error"),
        }
        
        return {
            "raw_content": content[:200],
            "parsed": parsed,
            "analysis": analysis,
            "time": result.get("time"),
        }
        
    except Exception as e:
        return {"error": str(e)}


def explore_model(model: str, timeout: int = 30):
    """Run all tests on a model and report results."""
    family = get_model_family(model)
    print(f"\n{'='*60}")
    print(f"Exploring model: {model} (family: {family})")
    print(f"{'='*60}")
    
    results = {}
    
    for test_name in TEST_PROMPTS:
        print(f"\n--- Test: {test_name} ---")
        result = test_model(model, test_name, timeout=timeout)
        
        analysis = result.get("analysis", {})
        
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            status = "ERROR"
        elif analysis.get("has_json"):
            print(f"  ✅ JSON parsed: {analysis.get('first_keys', [])[:3]}")
            status = "PASS"
        elif analysis.get("has_json_chars"):
            print(f"  ⚠️ Has JSON but not parsed: {analysis.get('content_first_50', '')[:40]}")
            status = "PARTIAL"
        else:
            print(f"  ❌ No JSON: {analysis.get('content_first_50', '')[:40]}")
            status = "FAIL"
            
        results[test_name] = {
            "status": status,
            "result": result,
        }
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for {model}:")
    print(f"{'='*60}")
    
    for test_name, data in results.items():
        status = data.get("status", "ERROR")
        print(f"  {status:8} - {test_name}")
    
    # Recommend best approach
    passed = [t for t, r in results.items() if r.get("status") == "PASS"]
    if passed:
        print(f"\n✅ Best approach: {passed[0]}")
        return passed[0]
    else:
        partial = [t for t, r in results.items() if r.get("status") == "PARTIAL"]
        if partial:
            print(f"\n⚠️ Try: {partial[0]}")
            return partial[0]
    
    return None


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "foundation"
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    best = explore_model(model, timeout)
    print(f"\n🎯 Recommended approach: {best}")