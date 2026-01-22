# Prompt Optimization Directive

## Purpose
This directive defines the 3-phase workflow for testing and optimizing system prompts across 17 LLM models, using **sub-agents with tool support** and objective quality evaluation via DeepEval/LLM-as-judge.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Prompt Optimiser Agent (Orchestrator)                                       │
│   Receives: system_prompt + user_prompts + tools.yaml + evaluation_criteria │
│   Confirms: model selection with user                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐   ...
│ Sub-Agent │ │ Sub-Agent │ │ Sub-Agent │   (17 models)
│ (Claude)  │ │ (Gemini)  │ │ (Grok)    │
│           │ │           │ │           │
│ system_   │ │ system_   │ │ system_   │
│ prompt +  │ │ prompt +  │ │ prompt +  │
│ tools.yml │ │ tools.yml │ │ tools.yml │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      │             │             │
      ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Each Sub-Agent Can:                                                         │
│   - Receive the user prompt                                                │
│   - Call tools (RAG search, knowledge graph, etc.)                         │
│   - Execute multi-step reasoning with tool results                         │
│   - Return final response                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Metrics Collected Per Sub-Agent:                                            │
│   - Total latency (end-to-end execution time)                              │
│   - LLM latency (time spent on model calls only)                           │
│   - Tool latency (time spent executing tools)                              │
│   - Tool calls (which tools, how many, in what sequence)                   │
│   - Tokens (input/output)                                                  │
│   - Cost (USD)                                                             │
│   - Number of iterations (agent loop cycles)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DeepEval Evaluation                                                         │
│   - Judge output quality against criteria                                  │
│   - Score instruction following, helpfulness, etc.                         │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Results Table                                                               │
│   Model | Quality | Latency | Tool Calls | Cost | Tokens | Reliability     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Execution Scripts

| File | Purpose |
|------|---------|
| `execution/sub_agent.py` | SubAgent class with tool-calling capability |
| `execution/agent_executor.py` | Parallel execution of sub-agents |
| `execution/prompt_tester.py` | PromptTester with both modes (API + Agent) |
| `execution/prompt_tester_schema.py` | Data models with tool call tracking |
| `execution/deepeval_evaluator.py` | Quality scoring |
| `config/models.yaml` | Model definitions and pricing |
| `config/tools.yaml` | Tool definitions for sub-agents |

---

## Two Execution Modes

### Mode 1: Direct API Calls (No Tools)
Use when your system prompt doesn't require tool usage.

```python
from execution.prompt_tester import PromptTester

tester = PromptTester()
summary = await tester.run_phase1_screening(
    system_prompt="You are a helpful assistant.",
    test_prompts=test_prompts,
)
```

### Mode 2: Sub-Agents with Tools (Recommended)
Use when your system prompt instructs models to use tools.

```python
from execution.prompt_tester import PromptTester

tester = PromptTester()
summary = await tester.run_agent_phase1_screening(
    system_prompt="You are a RAG assistant. Use the search tool to find relevant information.",
    test_prompts=test_prompts,
    tools_config_path="config/tools.yaml",
)
```

---

## Phase 1: Initial Screening (Sub-Agent Mode)

**Goal**: Quickly identify which models perform well as agents with tool usage.

### Execution

```python
from execution.prompt_tester import PromptTester
from execution.prompt_tester_schema import TestPrompt, EvaluationCriteria
from execution.deepeval_evaluator import get_evaluator

# Initialize
tester = PromptTester()
evaluator = get_evaluator(use_deepeval=True)

# Define test prompts
test_prompts = [
    TestPrompt(
        id="t1",
        content="What AI projects has Pierre built?",
        category="rag_retrieval"
    ),
    TestPrompt(
        id="t2",
        content="Find connections between RAG and vector databases",
        category="knowledge_graph"
    ),
    TestPrompt(
        id="t3",
        content="Summarize my recent work history",
        category="memory_retrieval"
    ),
]

# Define evaluation criteria
evaluation_criteria = [
    EvaluationCriteria(
        name="Instruction Following",
        criteria="How well does the response follow the system prompt instructions?",
        threshold=0.7
    ),
    EvaluationCriteria(
        name="Tool Usage",
        criteria="Did the agent use appropriate tools to answer the question?",
        threshold=0.7
    ),
    EvaluationCriteria(
        name="Response Quality",
        criteria="Is the final response helpful, accurate, and well-structured?",
        threshold=0.7
    ),
]

# Run Phase 1 with sub-agents
summary = await tester.run_agent_phase1_screening(
    system_prompt="""You are a RAG assistant with access to a knowledge base.
    Use the dynamic_hybrid_search tool to find relevant information.
    Always cite your sources.""",
    test_prompts=test_prompts,
    tools_config_path="config/tools.yaml",
    num_prompts=3,
    evaluation_criteria=evaluation_criteria,
    max_iterations=10,  # Max agent loop cycles
)

# Evaluate responses
summary = await evaluator.evaluate_test_run(summary)

# View results
print(tester.format_summary_report(summary))
print(f"Top models: {summary.get_top_models(4)}")
```

### Metrics Captured

| Metric | Description | Source |
|--------|-------------|--------|
| `total_latency_ms` | End-to-end execution time | SubAgent timer |
| `llm_latency_ms` | Time spent on LLM calls | SubAgent timer |
| `tool_latency_ms` | Time spent executing tools | SubAgent timer |
| `num_tool_calls` | Number of tool invocations | SubAgent counter |
| `tools_used` | List of tools called | SubAgent tracker |
| `num_iterations` | Agent loop cycles | SubAgent counter |
| `input_tokens` | Prompt tokens used | OpenRouter API |
| `output_tokens` | Response tokens | OpenRouter API |
| `cost_usd` | Calculated cost | Model pricing |
| `eval_scores` | Quality scores | DeepEval evaluator |

### Expected Cost (Sub-Agent Mode)

Sub-agents may use more tokens due to multi-turn conversations:
- Model responses: ~$0.10-1.00 for 51 executions (17 models × 3 prompts)
- Tool executions: Depends on your tool endpoints
- Evaluation: ~$0.02-0.10 for 51 evaluations
- **Total Phase 1**: ~$0.15-1.50

---

## Phase 2: Model Selection

**Goal**: Analyze Phase 1 results and select top 2-4 models for deep testing.

### Evaluation Criteria Weights

| Factor | Weight | Rationale |
|--------|--------|-----------|
| Quality Score | 40% | Does it follow instructions and produce good output? |
| Latency | 25% | User experience and throughput |
| Cost | 20% | Budget constraints |
| Reliability | 15% | Success rate and consistency |

### Analysis Code

```python
# Rankings are automatically calculated
rankings = summary.model_rankings

# Print detailed comparison
for r in rankings[:6]:
    print(f"{r.model_id}:")
    print(f"  Quality: {r.avg_quality_score:.2f}")
    print(f"  Latency: {r.avg_latency_ms:.0f}ms (LLM: {r.avg_llm_latency_ms:.0f}ms, Tools: {r.avg_tool_latency_ms:.0f}ms)")
    print(f"  Tool Calls: {r.avg_tool_calls:.1f} avg")
    print(f"  Tools Used: {', '.join(r.tools_used)}")
    print(f"  Cost: ${r.avg_cost_usd:.4f}")
    print(f"  Reliability: {r.success_rate:.0%}")
    print(f"  Composite: {r.composite_score:.3f}")

# Select top models
selected_models = summary.get_top_models(4)
```

### Decision Framework

Consider these patterns when selecting models:
- **High tool usage, high quality**: Model understands when to use tools
- **Low tool usage, high quality**: Model may not need tools OR doesn't know how to use them
- **High latency, many tool calls**: Model over-uses tools or gets stuck in loops
- **Low cost, good quality**: Efficient model for budget optimization

---

## Phase 3: Deep Testing (Sub-Agent Mode)

**Goal**: Comprehensive testing of selected models with full prompt suite.

### Execution

```python
# Use models selected from Phase 2
selected_models = summary.get_top_models(4)

# Full test prompt suite
all_prompts = [
    TestPrompt(id="t1", content="What AI projects has Pierre built?"),
    TestPrompt(id="t2", content="Find connections between RAG and vector databases"),
    # ... 10-20 more prompts covering all use cases
]

# Run Phase 3 with sub-agents
deep_summary = await tester.run_agent_phase3_deep_testing(
    system_prompt=system_prompt,
    test_prompts=all_prompts,
    model_ids=selected_models,
    tools_config_path="config/tools.yaml",
    evaluation_criteria=evaluation_criteria,
    max_iterations=10,
)

# Evaluate
deep_summary = await evaluator.evaluate_test_run(deep_summary)

# Final report
print(tester.format_summary_report(deep_summary))
```

---

## Tool Configuration

Tools are defined in `config/tools.yaml`. Sub-agents receive these tools as function calling schema.

### Parameter Sources

| Source | Description |
|--------|-------------|
| `model` | Parameter provided by the LLM during tool call |
| `sub_agent` | Parameter injected by SubAgent (e.g., session_id) |
| `env` | Parameter loaded from environment variable |
| `prompt_optimiser` | Parameter provided by parent orchestrator |

### Example Tool Definition

```yaml
tools:
  - name: dynamic_hybrid_search
    description: |
      Query data from knowledgebase using hybrid search.
    parameters:
      query:
        type: string
        required: true
        source: model  # LLM decides what to search
      session_id:
        type: string
        required: true
        source: sub_agent  # Auto-injected by SubAgent
      dense_weight:
        type: number
        default: 0.5
        source: model
    endpoint:
      url: "${N8N_WEBHOOK_BASE_URL}/webhook-id"
      method: POST
```

---

## Results Table Format

The final results table includes tool-related metrics:

| Model | Quality | Latency | LLM Time | Tool Time | Tool Calls | Cost | Reliability |
|-------|---------|---------|----------|-----------|------------|------|-------------|
| claude-sonnet-4.5 | 0.92 | 2340ms | 1800ms | 540ms | 2.3 | $0.0045 | 100% |
| gemini-2.5-flash | 0.88 | 1560ms | 1200ms | 360ms | 1.8 | $0.0012 | 100% |
| grok-4-fast | 0.85 | 890ms | 650ms | 240ms | 1.2 | $0.0008 | 95% |

---

## Error Handling

| Error | Likely Cause | Resolution |
|-------|--------------|------------|
| `Max iterations reached` | Agent stuck in tool loop | Increase `max_iterations` or simplify system prompt |
| `Tool execution failed` | Tool endpoint error | Check tool endpoint connectivity and credentials |
| `HTTP 429` | Rate limit exceeded | Reduce `max_concurrent` parameter |
| `No choices in response` | Model error | Check model availability on OpenRouter |

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

## Learnings

_This section is updated as the system is used and edge cases are discovered._

<!-- Add learnings here as they are discovered -->
