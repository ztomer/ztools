#!/usr/bin/env python3
"""
Model Quirks Explorer - Systematically probe model behaviors.
Used to discover prompt/response patterns for different models.

Run: python3 explore_model_quirks.py <model>
"""

import sys
import json
import argparse
from lib.osaurus_lib import call, extract_json, filter_json_items, fix_json_years
from lib.validators_lib import check_source_extraction, has_item_details

TEST_PROMPTS = {
    "simple_json": {
        "system": "Output JSON with name and age.",
        "user": "My name is John, I am 25 years old",
    },
    "no_preamble": {
        "system": "Output JSON now. No explanations. No markdown.",
        "user": "Events: Spring Festival, Indoor Coding, Farmers Market",
    },
    "schema_strict": {
        "system": "Output JSON: [{\"name\": \"...\", \"location\": \"...\"}]",
        "user": "Spring Festival at Parks, Indoor Coding at Library, Farmers Market",
    },
    "structured": {
        "system": "Output ONLY valid JSON array. Each item must have: name, location, target_ages, price, weather, day. No text before or after.",
        "user": "Spring Festival: Downsview Park, all ages, free. Coding: Library, 8-14, $20. Farmers Market: Maple Village, all ages, free.",
    },
    "ultra_strict": {
        "system": "CRITICAL: Output ONLY a JSON array. Nothing else. No intro text. No markdown. Format: [{\"name\":\"value\",\"location\":\"value\"}]",
        "user": "Extract to JSON: Spring Festival at Downsview Park, Indoor Coding at Library, Farmers Market at Maple Village",
    },
}


def run_test(model: str, test_name: str, timeout: int = 30) -> dict:
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
            parse_json=True,
            timeout=timeout,
        )
        
        content = result.get("content", "")
        parsed = result.get("parsed")
        
        analysis = {
            "has_json": bool(parsed),
            "has_markdown": "**" in content or "```" in content,
            "has_table": "|" in content,
            "item_count": len(parsed) if isinstance(parsed, list) else 0,
            "has_details": sum(1 for p in (parsed or []) if has_item_details(p)) if isinstance(parsed, list) else 0,
        }
        
        return {
            "content": content[:500],
            "parsed": parsed,
            "analysis": analysis,
            "time": result.get("time"),
        }
        
    except Exception as e:
        return {"error": str(e)}


def test_source_matching(model: str, timeout: int = 60) -> dict:
    """Test if model extracts from input (not hallucinated)."""
    system = "Output JSON now. Extract from the context provided."
    user = """High-Signal Transient Events:
- Spring Festival at Downsview Park: Outdoor rides. April 20-22. All ages.
- Indoor Coding Workshop: Learn Python. April 21. Ages 8-14.
- Farmers Market: Fresh produce. April 20. All ages."""
    
    try:
        result = call(model=model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], task="json", parse_json=True, timeout=timeout)
        parsed = result.get("parsed", [])
        
        if not parsed:
            return {"status": "FAIL", "reason": "no parsed items"}
        
        ratio = check_source_extraction(parsed, user)
        
        # Test our fixes
        filtered = filter_json_items(parsed)
        fixed_years = fix_json_years(parsed)
        
        # Check details
        with_details = sum(1 for p in parsed if has_item_details(p)) if isinstance(parsed, list) else 0
        
        return {
            "status": "PASS" if ratio >= 0.5 else "FAIL",
            "source_match_ratio": ratio,
            "items": len(parsed),
            "with_details": with_details,
            "filtered_count": len(filtered),
            "raw_sample": str(parsed[:2])[:200],
        }
        
    except Exception as e:
        return {"status": "ERROR", "reason": str(e)}


def explore_model(model: str, timeout: int = 30):
    """Run tests on a model and report results."""
    print(f"\n{'='*60}")
    print(f"Exploring model: {model}")
    print(f"{'='*60}")
    
    results = {}
    
    for test_name in TEST_PROMPTS:
        print(f"\n--- Test: {test_name} ---")
        result = run_test(model, test_name, timeout=timeout)
        
        analysis = result.get("analysis", {})
        
        if result.get("error"):
            print(f"  ERROR: {result['error']}")
            status = "ERROR"
        elif analysis.get("has_json"):
            details = analysis.get("has_details", 0)
            items = analysis.get("item_count", 0)
            print(f"  PASS: {items} items, {details} with details")
            status = "PASS" if details > 0 else "PARTIAL"
        elif analysis.get("has_markdown"):
            print(f"  WARN: Has markdown in output")
            status = "PARTIAL"
        else:
            print(f"  FAIL: No JSON parsed")
            status = "FAIL"
            
        results[test_name] = {"status": status, "result": result}
    
    # Source matching test
    print(f"\n--- Source Matching Test ---")
    src_result = test_source_matching(model, timeout=90)
    print(f"  Status: {src_result.get('status')}")
    print(f"  Match ratio: {src_result.get('source_match_ratio', 0):.0%}")
    print(f"  Items: {src_result.get('items', 0)}, with details: {src_result.get('with_details', 0)}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Results for {model}:")
    for test_name, data in results.items():
        status = data.get("status", "ERROR")
        print(f"  [{status}] {test_name}")
    print(f"  [SOURCE] {src_result.get('status')} (match: {src_result.get('source_match_ratio', 0):.0%})")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore model quirks")
    parser.add_argument("model", nargs="?", default="foundation", help="Model to test")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout per test")
    args = parser.parse_args()
    
    explore_model(args.model, args.timeout)