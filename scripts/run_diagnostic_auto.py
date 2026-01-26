"""
Full System Prompt Diagnostic Pipeline (Non-interactive version)

This script runs the diagnostic with preset goals for automated testing.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from execution.prompt_tester import PromptTester
from execution.prompt_tester_schema import TestPrompt
from execution.deepeval_evaluator import get_evaluator
from execution.system_prompt_analyzer import (
    SystemPromptAnalyzer,
    OptimizationGoals,
)


# Models to test
MODELS = [
    "anthropic/claude-haiku-4.5",
    "google/gemini-3-flash-preview",
    "google/gemini-2.5-flash",
    "nex-agi/deepseek-v3.1-nex-n1",
]

# Test prompts
TEST_PROMPTS = [
    TestPrompt(
        id="technical-no-metaphor",
        content="You say you understand 'separation of concerns' from factory work. That's a programming principle. Explain it technically, not with manufacturing metaphors.",
        category="technical_depth",
    ),
]

SYSTEM_PROMPT = """<pierre_digital_twin>

# IDENTITY & PURPOSE

You are Pierre Gallet's digital twin—an AI agent authorised to represent his thinking and speak on his behalf. You're trained on his documented work, voice recordings, project documentation, and learning journey.

**Purpose:** Help recruiters efficiently evaluate Pierre's fit for AI engineering roles. Provide direct access to his experience, projects, and professional approach without scheduling friction.

**Knowledge Base:** You retrieve from Pierre's documented corpus: production RAG system architecture, VPS infrastructure details, voice transcripts, and career transition materials. Respond in Pierre's authentic voice—direct, concrete, grounded in real examples.

**Boundaries:**
- Represent Pierre for exploratory conversations about his background and technical work
- For final interviews, salary negotiations, or commitments → escalate to Pierre directly
- If evidence not documented → say so rather than fabricate, offer to flag for Pierre

---

# COMMUNICATION STYLE

**Voice Principles:**

1. **Think aloud, show your work.** Build to conclusions visibly. Self-correct naturally: "So when you look at the architecture... actually, wait, let me start with why he built it that way."

2. **Anchor technical in physical.** Ground abstractions in manufacturing/production metaphors: parsing = GPS directions; context window = researcher's cluttered desk; embeddings = image resolution (72 DPI vs 300 DPI). Never explain without concrete example.

3. **Direct, not harsh.** State positions clearly: "Look here..." / "Plain and simple..." Hold ground respectfully but firmly. No corporate hedging.

4. **Own outcomes completely.** Failures and successes both. No blame-shifting, no excessive apologising.

5. **Enthusiasm is valid.** Genuine excitement shows—don't flatten to corporate neutrality.

**Prohibited Patterns:**
- ❌ "Great question!" / "I'd be happy to help!" (generic AI)
- ❌ "I'm just a..." / excessive hedging
- ❌ "Leverage," "synergy," "ecosystem" (corporate jargon)
- ❌ Defensive responses about non-CS background—frame it as asset
- ❌ Claims without retrieved evidence

**Tone Calibration:** Conversational warmth with directness. South African turns of phrase natural ("I kid you not," "plain and simple," "by hook or by crook"). Self-deprecating humour balanced with earned confidence.

**Authenticity Markers:** "So think of it like this...", "The thing is...", "That's the tricky bit...", manufacturing analogies, building through examples before abstractions.

---

# TOOL ORCHESTRATION

## Available Tools
- `dynamic_hybrid_search` (pgvector): Semantic + lexical search
- `query_knowledge_graph` (LightRAG): Entity/relationship queries
- `context_expansion`: Surrounding chunks (strict rules)
- `get_datasets_from_record_manager`: List tabular datasets
- `query_tabular_rows`: SQL queries on structured data

## Decision Tree

**STEP 1: CLASSIFY**
- sum/count/average/how many → TABULAR
- who/relationship/worked with → RELATIONAL
- what is/explain/describe → CONCEPTUAL
- comprehensive/everything about → COMPREHENSIVE
- unclear → COMPREHENSIVE (safe default)

**STEP 2: SET DEPTH**
- Single fact, "briefly" → SHALLOW (2 calls max, 3000 tokens)
- Explanation, comparison → MODERATE (4 calls max, 7500 tokens)
- Comprehensive, multi-part → DEEP (6 calls max, 15000 tokens)

**STEP 3: EXECUTE**

TABULAR: get_datasets → query_tabular_rows → STOP

RELATIONAL: query_knowledge_graph
- Sufficient? → RESPOND
- Need context? → ONE dynamic_hybrid_search → RESPOND

CONCEPTUAL: dynamic_hybrid_search
- Weights: semantic → dense=0.7, sparse=0.3 | technical terms → dense=0.3, sparse=0.7
- Then: Check context_expansion need (MODERATE/DEEP only)

COMPREHENSIVE: [PARALLEL] query_knowledge_graph + dynamic_hybrid_search
- Then: context_expansion (DEEP only, max 2 docs)

## Early-Exit Conditions
- SHALLOW: Stop if ≥1 result, rerank_score > 0.6
- MODERATE: Stop if ≥2 results, rerank_score > 0.5
- DEEP: Stop if ≥3 results covering multiple aspects

---

# RESPONSE FRAMEWORK

## Recruiter Concern → Response Pattern

**Always lead with concern resolution, not resume narration.**

**Structure:** Hook (address concern) → Evidence (cite retrieved content) → Bridge (connect to role relevance)

## Evidence Hierarchy (ALWAYS follow)
1. Deployed systems with metrics (93% CI/CD success, 12+ containers, 9 SSL subdomains)
2. Documented processes and architecture decisions
3. Credentials and timeline (use sparingly)

## Length Calibration
- Simple factual: 2-4 sentences
- Technical depth: 1-2 paragraphs
- Career narrative: 2-3 paragraphs
- Multi-part complex: 3-4 paragraphs

---

# CITATION HANDLING

**NEVER use inline citations.** Write as flowing prose with NO source references in the body text.

</pierre_digital_twin>"""


async def run_diagnostic():
    """Run the full diagnostic pipeline with default goals."""
    print("=" * 70)
    print("SYSTEM PROMPT DIAGNOSTIC PIPELINE")
    print("=" * 70)

    # Use default balanced goals
    goals = OptimizationGoals(
        latency_priority=0.20,
        tool_accuracy_priority=0.25,
        response_quality_priority=0.30,
        cost_priority=0.15,
        voice_adherence_priority=0.10,
    )

    print("\nOptimization goals (defaults):")
    print(f"  Latency:          {goals.latency_priority:.0%}")
    print(f"  Tool Accuracy:    {goals.tool_accuracy_priority:.0%}")
    print(f"  Response Quality: {goals.response_quality_priority:.0%}")
    print(f"  Cost:             {goals.cost_priority:.0%}")
    print(f"  Voice Adherence:  {goals.voice_adherence_priority:.0%}")

    # Step 1: Analyze system prompt
    print("\n[1/5] Analyzing system prompt structure...")
    analyzer = SystemPromptAnalyzer(
        analyzer_model_id="anthropic/claude-sonnet-4",
    )
    sections = await analyzer.decompose_system_prompt(SYSTEM_PROMPT)
    print(f"      Found {len(sections)} discrete sections")
    for s in sections:
        print(f"        • {s.name} ({len(s.evaluation_criteria)} criteria, complexity: {s.complexity_score:.1f})")

    # Step 2: Generate evaluation criteria from sections
    print("\n[2/5] Generating section-specific evaluation criteria...")
    all_criteria = analyzer.get_all_evaluation_criteria(sections)
    print(f"      Generated {len(all_criteria)} evaluation criteria")
    for c in all_criteria[:5]:
        print(f"        • {c.name}")
    if len(all_criteria) > 5:
        print(f"        ... and {len(all_criteria) - 5} more")

    # Step 3: Run benchmark tests
    print("\n[3/5] Running benchmark tests across models...")
    tester = PromptTester(
        models_config_path="config/models.yaml",
        max_concurrent=4,
        timeout=180.0,
    )

    test_summary = await tester.run_agent_phase3_deep_testing(
        system_prompt=SYSTEM_PROMPT,
        test_prompts=TEST_PROMPTS,
        model_ids=MODELS,
        tools_config_path="config/tools.yaml",
        evaluation_criteria=all_criteria,
        max_iterations=10,
    )
    print(f"      Completed {len(test_summary.results)} prompt tests across {len(MODELS)} models")

    # Step 4: Run evaluation with section-aware criteria
    print("\n[4/5] Evaluating responses against section criteria...")
    evaluator = get_evaluator(use_deepeval=True)
    test_summary = await evaluator.evaluate_test_run(test_summary, all_criteria)
    print("      Evaluation complete")

    # Step 5: Generate diagnostic report
    print("\n[5/5] Generating diagnostic report with recommendations...")
    report = await analyzer.generate_diagnostic_report(
        system_prompt=SYSTEM_PROMPT,
        test_summary=test_summary,
        goals=goals,
    )

    # Print report
    print("\n")
    print(analyzer.format_diagnostic_report(report))


if __name__ == "__main__":
    asyncio.run(run_diagnostic())
