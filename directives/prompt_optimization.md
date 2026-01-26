# Prompt Optimization Directive

## Purpose

This directive defines the workflow for **analyzing, testing, and optimizing system prompts** using LLM-powered decomposition, multi-model benchmarking, and diagnostic reasoning.

The system answers: **What parts of my system prompt are working, what's broken, and how do I fix it?**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SYSTEM PROMPT ANALYSIS                                                   │
│    SystemPromptAnalyzer decomposes prompt into testable sections            │
│    Generates evaluation criteria per section automatically                  │
│    Identifies potential confusion points and latency impacts                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. USER GOAL SETTING                                                        │
│    User specifies optimization priorities (weights):                        │
│    - Latency (faster responses)                                             │
│    - Tool Accuracy (correct tool usage per decision tree)                   │
│    - Response Quality (helpful, accurate answers)                           │
│    - Cost Efficiency (lower token usage)                                    │
│    - Voice Adherence (authentic voice/style)                                │
│    - Custom Goals (user-specified)                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. BENCHMARK EXECUTION                                                      │
│    Sub-agents with tool support across multiple models                      │
│    Granular metric collection per execution                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
     ┌───────────┐           ┌───────────┐           ┌───────────┐
     │ Sub-Agent │           │ Sub-Agent │           │ Sub-Agent │
     │ (Claude)  │           │ (Gemini)  │           │ (Grok)    │
     │           │           │           │           │           │
     │ Metrics:  │           │ Metrics:  │           │ Metrics:  │
     │ -latency  │           │ -latency  │           │ -latency  │
     │ -tools    │           │ -tools    │           │ -tools    │
     │ -tokens   │           │ -tokens   │           │ -tokens   │
     │ -iters    │           │ -iters    │           │ -iters    │
     └───────────┘           └───────────┘           └───────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. SYSTEM-PROMPT-AWARE EVALUATION                                           │
│    Judge receives FULL system prompt as context                             │
│    Per-section adherence scoring                                            │
│    Tool relevancy evaluated against decision tree rules                     │
│    Voice/style compliance checked against specified patterns                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. DIAGNOSTIC REPORT                                                        │
│    Section-by-section adherence scores                                      │
│    Issues with evidence from actual responses                               │
│    Confusion indicators (loops, wrong tools, off-voice)                     │
│    Before/after recommendations with examples                               │
│    Estimated improvement potential                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Scripts

| File | Purpose |
|------|---------|
| `execution/system_prompt_analyzer.py` | LLM-powered prompt decomposition and diagnostics |
| `execution/sub_agent.py` | SubAgent class with tool-calling capability |
| `execution/agent_executor.py` | Parallel execution of sub-agents |
| `execution/prompt_tester.py` | PromptTester with both modes (API + Agent) |
| `execution/prompt_tester_schema.py` | Data models with granular metric tracking |
| `execution/deepeval_evaluator.py` | System-prompt-aware quality scoring |
| `config/models.yaml` | Model definitions and pricing |
| `config/tools.yaml` | Tool definitions for sub-agents |
| `scripts/run_diagnostic.py` | Interactive full diagnostic pipeline |
| `scripts/run_diagnostic_auto.py` | Non-interactive diagnostic with defaults |

---

## Full Diagnostic Pipeline

### Step 1: System Prompt Decomposition

The `SystemPromptAnalyzer` uses an LLM to decompose any system prompt into discrete, testable sections.

```python
from execution.system_prompt_analyzer import SystemPromptAnalyzer

analyzer = SystemPromptAnalyzer(
    analyzer_model_id="anthropic/claude-sonnet-4",  # Needs strong reasoning
)

sections = await analyzer.decompose_system_prompt(system_prompt)

# Each section contains:
# - name: "Tool Orchestration"
# - content: The actual text
# - purpose: What this section achieves
# - testable_behaviors: ["Classifies query type", "Follows decision tree"]
# - evaluation_criteria: Auto-generated criteria for scoring
# - potential_confusion_points: ["Complex decision tree", "Unclear priority"]
# - latency_impact: "high"
# - complexity_score: 0.8
```

**Example Output:**
```
Found 11 discrete sections:
  • Identity & Purpose (2 criteria, complexity: 0.3)
  • Voice Principles (3 criteria, complexity: 0.7)
  • Tool Classification Decision Tree (3 criteria, complexity: 0.8)
  • Tool Execution Patterns (3 criteria, complexity: 0.9)
  • Response Framework Structure (3 criteria, complexity: 0.7)
  ...
```

### Step 2: User Goal Setting

Users specify their optimization priorities as weights (must sum to 100%):

```python
from execution.system_prompt_analyzer import OptimizationGoals

goals = OptimizationGoals(
    latency_priority=0.20,        # 20% - faster responses
    tool_accuracy_priority=0.25,  # 25% - correct tool usage
    response_quality_priority=0.30,  # 30% - helpful answers
    cost_priority=0.15,           # 15% - lower token usage
    voice_adherence_priority=0.10,  # 10% - authentic voice
    custom_goals=["Avoid manufacturing metaphors when user requests technical-only"],
)
```

### Step 3: Generate Section-Specific Evaluation Criteria

Criteria are auto-generated from the decomposed sections:

```python
all_criteria = analyzer.get_all_evaluation_criteria(sections)
# Returns 20-30 criteria depending on system prompt complexity
```

### Step 4: Run Benchmark Tests

Execute sub-agents with tools across multiple models:

```python
from execution.prompt_tester import PromptTester
from execution.prompt_tester_schema import TestPrompt

tester = PromptTester(max_concurrent=4, timeout=180.0)

test_prompts = [
    TestPrompt(
        id="technical-depth",
        content="What's the architecture of your RAG system?",
        category="technical_depth",
    ),
    TestPrompt(
        id="career-background",
        content="You don't have a CS degree - why should I trust you?",
        category="career_background",
    ),
]

summary = await tester.run_agent_phase3_deep_testing(
    system_prompt=system_prompt,
    test_prompts=test_prompts,
    model_ids=["anthropic/claude-haiku-4.5", "google/gemini-2.5-flash"],
    tools_config_path="config/tools.yaml",
    evaluation_criteria=all_criteria,
    max_iterations=10,
)
```

### Step 5: System-Prompt-Aware Evaluation

The evaluator receives the **full system prompt as context**, enabling accurate adherence scoring:

```python
from execution.deepeval_evaluator import get_evaluator

evaluator = get_evaluator(use_deepeval=True)
summary = await evaluator.evaluate_test_run(summary, all_criteria)

# The judge now sees:
# - The system prompt (tool orchestration rules, voice principles, etc.)
# - The user input
# - The model's response
# - And scores adherence to each section
```

### Step 6: Generate Diagnostic Report

```python
report = await analyzer.generate_diagnostic_report(
    system_prompt=system_prompt,
    test_summary=summary,
    goals=goals,
)

print(analyzer.format_diagnostic_report(report))
```

---

## Diagnostic Report Format

The report provides actionable insights for each system prompt section:

```
======================================================================
SYSTEM PROMPT DIAGNOSTIC REPORT
======================================================================
Prompt Hash: 8a324a3c
Analyzed: 2026-01-23T14:11:52

OPTIMIZATION GOALS:
  Latency: 20%
  Tool Accuracy: 25%
  Response Quality: 30%
  Cost: 15%
  Voice Adherence: 10%

----------------------------------------------------------------------
OVERALL SCORES
----------------------------------------------------------------------
  instruction_following     [████████░░░░░░░░░░░░] 0.42
  tool_accuracy             [██████░░░░░░░░░░░░░░] 0.35
  response_quality          [████████████░░░░░░░░] 0.62
  latency_efficiency        [█████░░░░░░░░░░░░░░░] 0.26
  cost_efficiency           [██████████████████░░] 0.94

----------------------------------------------------------------------
GOAL ACHIEVEMENT
----------------------------------------------------------------------
  ✗ latency              26%
  ✗ tool_accuracy        35%
  ✓ response_quality     62%
  ✓ cost                 94%
  ✗ voice_adherence      42%

----------------------------------------------------------------------
SECTION-BY-SECTION DIAGNOSTICS
----------------------------------------------------------------------

🔴 Tool Execution Patterns (Adherence: 35%)
   Priority: CRITICAL
   Issues:
     • No tool execution visible in any responses
     • CONCEPTUAL queries not triggering dynamic_hybrid_search
     • Agents don't understand when to use which tools
   Confusion Indicators:
     ⚡ Complete tool orchestration failure
     ⚡ No retrieval from documented corpus
   Latency Impact: +3000ms
   Recommendation: Enforce tool execution: 'CONCEPTUAL queries MUST trigger
                   dynamic_hybrid_search. Show: "Searching Pierre's docs..."'
   Before:
     "Follows TABULAR workflow exactly..."
   After:
     "MANDATORY: CONCEPTUAL → dynamic_hybrid_search with semantic weights..."

🟠 Voice Principles (Adherence: 55%)
   Priority: HIGH
   Issues:
     • Manufacturing metaphors abandoned when users request 'technical only'
     • Missing 'think aloud' and self-correction patterns
   Recommendation: 'Manufacturing metaphors are not optional. When users ask
                   for "technical only", respond: "The manufacturing lens IS
                   the technical explanation..."'

----------------------------------------------------------------------
PRIORITY RECOMMENDATIONS
----------------------------------------------------------------------
  1. [CRITICAL] Tool Execution: Make tool usage mandatory and visible
  2. [CRITICAL] Knowledge Base: Enforce retrieval before any claims
  3. [HIGH] Voice Principles: Strengthen voice consistency under pressure
  4. [HIGH] Identity: Maintain character when challenged

----------------------------------------------------------------------
ESTIMATED IMPROVEMENT POTENTIAL
----------------------------------------------------------------------
  latency: -10700ms potential
  quality: +40% if critical issues fixed

======================================================================
```

---

## Metrics Captured

### Per-Response Metrics

| Metric | Description | Source |
|--------|-------------|--------|
| `total_latency_ms` | End-to-end execution time | SubAgent timer |
| `llm_latency_ms` | Time spent on LLM calls only | SubAgent timer |
| `tool_latency_ms` | Time spent executing tools | SubAgent timer |
| `num_tool_calls` | Number of tool invocations | SubAgent counter |
| `tools_used` | List of tools called | SubAgent tracker |
| `num_iterations` | Agent loop cycles | SubAgent counter |
| `input_tokens` | Prompt tokens used | OpenRouter API |
| `output_tokens` | Response tokens | OpenRouter API |
| `cost_usd` | Calculated cost | Model pricing |

### Per-Section Evaluation Scores

| Metric | Description |
|--------|-------------|
| `adherence_score` | 0-1 score for section compliance |
| `issues_found` | Specific violations with evidence |
| `confusion_indicators` | Loops, wrong tools, off-voice patterns |
| `latency_impact_ms` | Time overhead from this section |
| `token_overhead` | Extra tokens due to section complexity |

### Per-Tool-Call Metrics

| Metric | Description |
|--------|-------------|
| `tool_name` | Which tool was called |
| `parameters` | Arguments passed |
| `latency_ms` | Tool execution time |
| `success` | Whether call succeeded |
| `relevance_score` | 0-1 score for appropriateness |
| `relevance_reason` | Why the call was/wasn't appropriate |

---

## Quick Start: Run Full Diagnostic

```bash
# Interactive (asks for goals)
PYTHONPATH=. python scripts/run_diagnostic.py

# Non-interactive (uses defaults)
PYTHONPATH=. python scripts/run_diagnostic_auto.py
```

---

## Configuration Files

### models.yaml
17 curated models across 4 providers (Anthropic, Google, xAI, DeepSeek).

### tools.yaml
RAG tools including:
- `dynamic_hybrid_search` - Vector + lexical search
- `query_knowledge_graph` - Entity relationship queries
- `context_expansion` - Expand search results
- `short_term_memory` - Conversation history
- `get_long_term_memories` - Zep temporal memories

### .env
```
OPENROUTER_API_KEY=sk-or-v1-...
N8N_WEBHOOK_BASE_URL=https://...
SUPABASE_URL=https://...
SUPABASE_ANON_KEY=...
ZEP_API_KEY=...
ZEP_USER_ID=...
```

---

## Error Handling

| Error | Likely Cause | Resolution |
|-------|--------------|------------|
| `Max iterations reached` | Agent stuck in tool loop | Simplify decision tree or add explicit stop conditions |
| `Tool execution failed` | Tool endpoint error | Check tool endpoint connectivity and credentials |
| `HTTP 429` | Rate limit exceeded | Reduce `max_concurrent` parameter |
| `No choices in response` | Model error | Check model availability on OpenRouter |
| `Parse error in decomposition` | Complex system prompt structure | Simplify or add clear section headers |

---

## Optimization Workflow

### 1. Initial Analysis
Run diagnostic to identify critical issues.

### 2. Fix Critical Issues First
Focus on sections with priority "CRITICAL" - these block core functionality.

### 3. Apply Before/After Changes
Use the specific recommendations with before/after examples.

### 4. Re-run Diagnostic
Verify improvements and identify next priority issues.

### 5. Iterate
Continue until goal achievement metrics are satisfactory.

---

## Learnings

_This section is updated as the system is used and edge cases are discovered._

### 2026-01-23: Initial Implementation
- Evaluator must receive system prompt as context for meaningful scoring
- Tool orchestration sections are frequently the source of "critical" issues
- Voice/style adherence degrades under user pressure (e.g., "explain technically, no metaphors")
- LLM decomposition works well for structured prompts with clear sections

### Key Insight: The Diagnostic Loop
The most effective workflow is:
1. Decompose → Test → Diagnose → Fix critical issues → Re-test
2. Focus on one section at a time
3. Use before/after examples from the report as templates
4. Track improvement across runs by comparing adherence scores
