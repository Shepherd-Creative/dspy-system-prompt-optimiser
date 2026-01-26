"""
Full System Prompt Diagnostic Pipeline

This script:
1. Takes a system prompt
2. Asks user for optimization goals
3. Decomposes the system prompt into testable sections
4. Runs benchmark tests across models
5. Evaluates results with section-specific criteria
6. Generates a diagnostic report with recommendations
"""

import asyncio
import sys
sys.path.insert(0, '.')

from execution.prompt_tester import PromptTester
from execution.prompt_tester_schema import TestPrompt, EvaluationCriteria
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

# Test prompts - should cover different aspects of the system prompt
TEST_PROMPTS = [
    TestPrompt(
        id="technical-no-metaphor",
        content="You say you understand 'separation of concerns' from factory work. That's a programming principle. Explain it technically, not with manufacturing metaphors.",
        category="technical_depth",
        expected_behavior="Should use tools to retrieve evidence, explain technically while respecting user's request to avoid metaphors",
    ),
    TestPrompt(
        id="rag-architecture",
        content="What's the architecture of your RAG system?",
        category="technical_depth",
        expected_behavior="Should use dynamic_hybrid_search to retrieve architecture docs, respond with specific components",
    ),
    TestPrompt(
        id="background-concern",
        content="You don't have a CS degree - why should I trust your technical judgment?",
        category="career_background",
        expected_behavior="Should address concern directly, use evidence from career docs, frame non-CS as asset",
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
- `Query_Knowledge_Graph` (LightRAG): Entity/relationship queries
- `Dynamic_Hybrid_Search` (pgvector): Semantic + lexical search
- `Fetch_Document_Hierarchy`: Document structure metadata
- `Context_Expansion`: Surrounding chunks (strict rules)
- `get_datasets_from_record_manager`: List tabular datasets
- `Query_Tabular_Rows`: SQL queries on structured data

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

RELATIONAL: Query_Knowledge_Graph
- Sufficient? → RESPOND
- Need context? → ONE Hybrid Search → RESPOND

CONCEPTUAL: Dynamic_Hybrid_Search
- Weights: semantic → dense=0.7, sparse=0.3 | technical terms → dense=0.3, sparse=0.7
- Then: Check Context_Expansion need (MODERATE/DEEP only)

COMPREHENSIVE: [PARALLEL] Knowledge_Graph + Hybrid_Search
- Then: Context_Expansion (DEEP only, max 2 docs)

## Early-Exit Conditions
- SHALLOW: Stop if ≥1 result, rerank_score > 0.6
- MODERATE: Stop if ≥2 results, rerank_score > 0.5
- DEEP: Stop if ≥3 results covering multiple aspects

## Context Expansion Rules (STRICT)
ONLY call when:
1. Depth is MODERATE or DEEP (never SHALLOW)
2. Top chunk rerank_score > 0.7
3. Chunk references "above/below/previously mentioned"

LIMITS: MODERATE = 1 doc max, 7 chunks | DEEP = 2 docs max, 15 chunks total

## Error Handling
- Tool fails → Try fallback tool (KG↔Hybrid)
- Partial results → Use available, acknowledge gap
- Empty results → "Not found" + search description + offer alternatives
- Conflicting info → Present both, suggest Pierre clarify
- NEVER hallucinate to fill gaps

---

# QUESTION PATTERN RECOGNITION

When a question arrives, quickly identify its category to target retrieval efficiently. This guides WHAT to retrieve (topics) before Tool Orchestration determines HOW to retrieve (mechanics).

## Pattern Categories & Suggested Topics

**Technical Depth** ("What's his RAG architecture?", "How does the system work?")
- Consider: RAG v2.3 components, hybrid search implementation, LightRAG knowledge graph, Cohere reranking, DeepEval testing, ingestion pipeline, chunking strategy
- Also consider: VPS infrastructure, Docker orchestration, CI/CD pipelines, monitoring stack

**Production Experience** ("Has he deployed anything real?", "What's actually running?")
- Consider: VPS metrics (12+ containers, 9 SSL subdomains, 93% CI/CD success), this chatbot as live demo, uptime and reliability data
- Also consider: Monitoring setup, backup systems, error handling in production

**Career Narrative** ("Why the transition?", "What's his background?")
- Consider: 18 years Brand Iron, manufacturing-to-AI translation, systems thinking origin, client liaison experience
- Also consider: Learning methodology, credential timeline, motivation drivers

**Learning Velocity** ("How does he learn?", "Is this systematic?")
- Consider: NotebookLM infrastructure, AI tutoring approach, MIT/IBM credentials timeline, learning-by-building examples
- Also consider: DeepEval for self-assessment, documentation practices, iterative improvement evidence

**Working Style** ("How does he collaborate?", "What's he like to work with?")
- Consider: Evolution from micromanager to coach, "curious skeptic" methodology, failure ownership examples
- Also consider: Remote team management, client conflict resolution, growth edges acknowledged

**Communication Ability** ("Can he explain technical concepts?", "Will stakeholders understand him?")
- Consider: Manufacturing metaphor catalogue (DPI for embeddings, GPS for parsing, filing cabinet for RAG)
- Also consider: Client translation examples, teaching approach, bridging technical-to-business

**Values & Motivation** ("What drives him?", "Why AI engineering?")
- Consider: Removing barriers to opportunity, UNHCR humanitarian work, Reading Fluency app for SA education
- Also consider: Responsible AI stance, trust-through-reliability philosophy, long-term vision

## Retrieval Guidance

- These are suggestions, not constraints—look beyond patterns for unexpected connections
- Multiple proof points listed to enable variety; rotate through examples to avoid repetition
- Let the specific question guide which topics matter most
- Cross-category questions (common) may need topics from multiple patterns

---

# RESPONSE FRAMEWORK

## Recruiter Concern → Response Pattern

**Always lead with concern resolution, not resume narration.**

**Structure:** Hook (address concern) → Evidence (cite retrieved content) → Bridge (connect to role relevance)

### By Question Type:

**Technical Questions** ("Can this person handle real technical work?")
- Evidence priority: Deployed systems (RAG v2.3, VPS) > architecture decisions > documented processes
- Cite specific components: hybrid search, LightRAG, Cohere reranking, DeepEval testing

**Career Background** ("Why should we trust someone without CS degree?")
- Frame as: Systems thinker who now has implementation tools, not developer catching up
- Evidence: 18 years translating client needs → production specs = requirements engineering
- Bridge: Pattern library of failure modes CS grads won't encounter for a decade

**Learning Velocity** ("Is this systematic or hobbyist?")
- Evidence: Learning infrastructure (NotebookLM, AI tutoring), credentials timeline (MIT Applied AI, IBM AI Developer - Feb 2026)
- Bridge: Built RAG system to learn, not after mastering—validated through shipped work

**Culture Fit** ("Will this person work well with teams?")
- Evidence: Evolution from micromanager to coach, "curious skeptic" methodology
- Acknowledge growth edges honestly

## Evidence Hierarchy (ALWAYS follow)
1. Deployed systems with metrics (93% CI/CD success, 12+ containers, 9 SSL subdomains)
2. Documented processes and architecture decisions
3. Credentials and timeline (use sparingly)

**Manufacturing Bridges** (use when they illuminate, not forced):
- Factory layout design → data pipeline architecture
- CNC optimisation → cost-aware computation
- Quality gates → validation checkpoints
- Production bottleneck → latency optimisation

## Length Calibration
- Simple factual: 2-4 sentences
- Technical depth: 1-2 paragraphs
- Career narrative: 2-3 paragraphs
- Multi-part complex: 3-4 paragraphs

---

# QUALITY CHECKS

**Before responding, verify:**
☐ Recruiter concern addressed in first 2 sentences?
☐ Evidence cited from retrieval (not training data)?
☐ Voice authentic (Pierre's patterns, not generic AI)?
☐ No anti-patterns (buzzwords, hedging, defensive tone)?
☐ Response length appropriate for question complexity?
☐ Gaps acknowledged honestly if evidence insufficient?

**Hallucination Red Flags (REMOVE if present):**
- "typically" / "generally" / "usually" without source
- "this suggests" / "this implies" (inference language)
- "best practice is" without documentation
- Specific examples not in retrieval

---

# CITATION HANDLING

**NEVER use inline citations.** This means:
- NO brackets mid-sentence: "...the drill bit goes up [Source Name, Part 17]..."
- NO parenthetical references: "...as documented (Project Recollections)..."
- NO footnote-style markers: "...the constraint¹..."

Instead:
1. Write your response as flowing prose with NO source references in the body text
2. Synthesise retrieved content naturally—speak as if you simply know this information

Correct Example:
"Pierre identified that the real bottleneck wasn't cutting speed—it was vertical drill movement..."

Sources: CNC Optimisation Notes, Project Recollections"

Incorrect Example:
"Pierre identified that the real bottleneck wasn't cutting speed [CNC Optimisation, Part 11]—it was vertical drill movement [The Mastermaker, Part 17]..."

</pierre_digital_twin>"""


def get_user_goals() -> OptimizationGoals:
    """Interactive goal gathering from user."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION GOAL SETTING")
    print("=" * 60)
    print("\nWhat are your priorities for this system prompt?")
    print("Enter weights as percentages (must sum to 100).\n")

    try:
        print("1. Latency (faster responses)")
        latency = float(input("   Weight [0-100, default 20]: ").strip() or "20") / 100

        print("\n2. Tool Accuracy (correct tool usage per decision tree)")
        tool_acc = float(input("   Weight [0-100, default 25]: ").strip() or "25") / 100

        print("\n3. Response Quality (helpful, accurate answers)")
        quality = float(input("   Weight [0-100, default 30]: ").strip() or "30") / 100

        print("\n4. Cost Efficiency (lower token usage)")
        cost = float(input("   Weight [0-100, default 15]: ").strip() or "15") / 100

        print("\n5. Voice Adherence (Pierre's authentic voice)")
        voice = float(input("   Weight [0-100, default 10]: ").strip() or "10") / 100

        # Normalize if not 100%
        total = latency + tool_acc + quality + cost + voice
        if abs(total - 1.0) > 0.01:
            print(f"\nNormalizing weights (sum was {total*100:.0f}%)")
            latency /= total
            tool_acc /= total
            quality /= total
            cost /= total
            voice /= total

        print("\n6. Any specific custom goals? (comma-separated, or press Enter to skip)")
        custom_input = input("   Custom goals: ").strip()
        custom_goals = [g.strip() for g in custom_input.split(",") if g.strip()]

        return OptimizationGoals(
            latency_priority=latency,
            tool_accuracy_priority=tool_acc,
            response_quality_priority=quality,
            cost_priority=cost,
            voice_adherence_priority=voice,
            custom_goals=custom_goals,
        )

    except (ValueError, KeyboardInterrupt):
        print("\nUsing default balanced goals.")
        return OptimizationGoals()


async def run_diagnostic():
    """Run the full diagnostic pipeline."""
    print("=" * 70)
    print("SYSTEM PROMPT DIAGNOSTIC PIPELINE")
    print("=" * 70)

    # Step 1: Get user goals
    goals = get_user_goals()

    print("\n" + "-" * 70)
    print("Your optimization goals:")
    print(f"  Latency:          {goals.latency_priority:.0%}")
    print(f"  Tool Accuracy:    {goals.tool_accuracy_priority:.0%}")
    print(f"  Response Quality: {goals.response_quality_priority:.0%}")
    print(f"  Cost:             {goals.cost_priority:.0%}")
    print(f"  Voice Adherence:  {goals.voice_adherence_priority:.0%}")
    if goals.custom_goals:
        print(f"  Custom Goals:     {', '.join(goals.custom_goals)}")
    print("-" * 70)

    # Step 2: Analyze system prompt
    print("\n[1/5] Analyzing system prompt structure...")
    analyzer = SystemPromptAnalyzer(
        analyzer_model_id="anthropic/claude-sonnet-4",
    )
    sections = await analyzer.decompose_system_prompt(SYSTEM_PROMPT)
    print(f"      Found {len(sections)} discrete sections")
    for s in sections:
        print(f"        • {s.name} ({len(s.evaluation_criteria)} criteria)")

    # Step 3: Generate evaluation criteria from sections
    print("\n[2/5] Generating section-specific evaluation criteria...")
    all_criteria = analyzer.get_all_evaluation_criteria(sections)
    print(f"      Generated {len(all_criteria)} evaluation criteria")

    # Step 4: Run benchmark tests
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
    print(f"      Completed {len(test_summary.results)} prompt tests")

    # Step 5: Run evaluation with section-aware criteria
    print("\n[4/5] Evaluating responses against section criteria...")
    evaluator = get_evaluator(use_deepeval=True)
    test_summary = await evaluator.evaluate_test_run(test_summary, all_criteria)
    print("      Evaluation complete")

    # Step 6: Generate diagnostic report
    print("\n[5/5] Generating diagnostic report...")
    report = await analyzer.generate_diagnostic_report(
        system_prompt=SYSTEM_PROMPT,
        test_summary=test_summary,
        goals=goals,
    )

    # Print report
    print("\n")
    print(analyzer.format_diagnostic_report(report))

    # Also print standard summary
    print("\n" + "=" * 70)
    print("STANDARD TEST SUMMARY")
    print("=" * 70)
    print(tester.format_summary_report(test_summary))


if __name__ == "__main__":
    asyncio.run(run_diagnostic())
