# LLM Debugging Playbook

## Critical Config

- **Osaurs server port: 1337** (not 8000!)
- Check with: `osaurus status`
- Health check: `curl http://localhost:1337/health`

## What Worked

### 1. Iterate with Real Runs
- Run 5-10 times after each fix to identify patterns vs noise
- Single runs are meaningless - patterns emerge from multiple runs
- Track success/failure rates to measure improvement

### 2. Server Restart Between Retries
- **Biggest single win** - Model outputs often garbage; restarting OAI server resets state
- Add `ensure_server()` call between LLM retries
- Wait 2-3s after restart before next call

### 3. Increase Retry Count + Delay
- Default 3 retries often insufficient for flaky models
- Increased to 5 retries with 2-3s delay between attempts
- Combined with server restart = much higher reliability

### 4. Debug Output That Matters
- Log which call # succeeded/failed
- Print raw LLM output before parsing
- Show item counts after each attempt

### 5. Graceful Failure Handling
- Return partial results instead of crashing
- Exit cleanly so scripts can be re-run
- Report what worked and what didn't

### 6. Robust Parsing
- Handle multiple JSON key names (`fixed_activities` vs `year_round_fixed_activities`)
- Field mapping solves schema mismatches between model versions
- JSON extraction handles trailing garbage

## What Didn't Help

- Single-step fixes - always need multiple iterations
- Adding more prompts - the model either works or doesn't
- Complex retry logic - simpler with server restart

## Key Technique

```
LLM Call → Parse → If fails, restart server → Wait → Retry
```

The server restart is the critical step. Without it, retrying the same bad state.