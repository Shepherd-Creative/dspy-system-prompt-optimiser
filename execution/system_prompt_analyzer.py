"""
System Prompt Analyzer - Decomposes system prompts into testable components
and generates diagnostic reports with optimization recommendations.

This is the core intelligence layer that:
1. Analyzes any system prompt to identify discrete, testable sections
2. Gathers user optimization goals
3. Generates evaluation criteria per section
4. Reasons over test results to recommend improvements
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from dotenv import load_dotenv

from .prompt_tester_schema import (
    EvaluationCriteria,
    TestRunSummary,
    ModelResponse,
)

load_dotenv()


@dataclass
class SystemPromptSection:
    """A discrete section of a system prompt that can be evaluated."""
    name: str
    content: str
    purpose: str
    testable_behaviors: List[str]
    evaluation_criteria: List[EvaluationCriteria]
    potential_confusion_points: List[str]
    latency_impact: str  # "low", "medium", "high"
    complexity_score: float  # 0-1


@dataclass
class OptimizationGoals:
    """User's optimization priorities."""
    latency_priority: float = 0.25  # 0-1 weight
    tool_accuracy_priority: float = 0.25
    response_quality_priority: float = 0.25
    cost_priority: float = 0.15
    voice_adherence_priority: float = 0.10
    custom_goals: List[str] = field(default_factory=list)

    def to_weights_dict(self) -> Dict[str, float]:
        return {
            "latency": self.latency_priority,
            "tool_accuracy": self.tool_accuracy_priority,
            "response_quality": self.response_quality_priority,
            "cost": self.cost_priority,
            "voice_adherence": self.voice_adherence_priority,
        }


@dataclass
class SectionDiagnostic:
    """Diagnostic results for a single system prompt section."""
    section_name: str
    adherence_score: float  # 0-1
    issues_found: List[str]
    latency_impact_ms: float
    confusion_indicators: List[str]
    token_overhead: int
    recommendation: str
    priority: str  # "critical", "high", "medium", "low"
    before_example: Optional[str] = None
    after_example: Optional[str] = None


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a system prompt."""
    system_prompt_hash: str
    analyzed_at: datetime
    optimization_goals: OptimizationGoals
    sections: List[SystemPromptSection]
    section_diagnostics: List[SectionDiagnostic]
    overall_scores: Dict[str, float]
    goal_achievement: Dict[str, float]  # How well each goal is being met
    priority_recommendations: List[str]
    estimated_improvement: Dict[str, str]  # "latency": "-15%", etc.


class SystemPromptAnalyzer:
    """
    Analyzes system prompts using LLM reasoning to decompose them into
    testable components and generate optimization recommendations.
    """

    def __init__(
        self,
        analyzer_model_id: str = "anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        """
        Initialize the analyzer.

        Args:
            analyzer_model_id: Model to use for analysis (needs strong reasoning)
            api_key: OpenRouter API key
            timeout: Request timeout
        """
        self.model_id = analyzer_model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout
        self._base_url = "https://openrouter.ai/api/v1"

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

    async def _call_llm(self, system: str, user: str) -> str:
        """Make an LLM call and return the response."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "X-Title": "System Prompt Analyzer",
                },
                json={
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.2,  # Low temp for analytical tasks
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def decompose_system_prompt(
        self,
        system_prompt: str,
    ) -> List[SystemPromptSection]:
        """
        Analyze a system prompt and decompose it into testable sections.

        Args:
            system_prompt: The full system prompt to analyze

        Returns:
            List of SystemPromptSection objects
        """
        analysis_prompt = """You are an expert at analyzing AI system prompts to identify discrete, testable components.

Analyze the provided system prompt and decompose it into sections. For each section, identify:
1. The section name and content boundaries
2. Its core purpose
3. Specific testable behaviors (what can be measured/verified)
4. Evaluation criteria (how to score adherence)
5. Potential confusion points (where an LLM might misinterpret)
6. Latency impact (does this section add processing overhead?)
7. Complexity score (0-1, how complex are the instructions?)

Return your analysis as a JSON array with this structure:
```json
[
  {
    "name": "Section Name",
    "content": "The actual text of this section (abbreviated if long)",
    "purpose": "What this section is meant to achieve",
    "testable_behaviors": [
      "Behavior 1 that can be verified",
      "Behavior 2 that can be verified"
    ],
    "evaluation_criteria": [
      {
        "name": "Criteria Name",
        "criteria": "Detailed description of what to evaluate",
        "threshold": 0.7
      }
    ],
    "potential_confusion_points": [
      "Point where LLM might get confused"
    ],
    "latency_impact": "low|medium|high",
    "complexity_score": 0.5
  }
]
```

Focus on sections that directly impact agent behavior, tool usage, response quality, and voice/style.
Be thorough - identify ALL discrete instructional components."""

        user_message = f"""Analyze this system prompt:

<system_prompt>
{system_prompt}
</system_prompt>

Return ONLY the JSON array, no other text."""

        response = await self._call_llm(analysis_prompt, user_message)

        # Parse JSON from response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            sections_data = json.loads(json_str)

            sections = []
            for s in sections_data:
                criteria = [
                    EvaluationCriteria(
                        name=c["name"],
                        criteria=c["criteria"],
                        threshold=c.get("threshold", 0.7),
                    )
                    for c in s.get("evaluation_criteria", [])
                ]

                section = SystemPromptSection(
                    name=s["name"],
                    content=s.get("content", ""),
                    purpose=s["purpose"],
                    testable_behaviors=s.get("testable_behaviors", []),
                    evaluation_criteria=criteria,
                    potential_confusion_points=s.get("potential_confusion_points", []),
                    latency_impact=s.get("latency_impact", "medium"),
                    complexity_score=s.get("complexity_score", 0.5),
                )
                sections.append(section)

            return sections

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Fallback: return a single section for the whole prompt
            return [
                SystemPromptSection(
                    name="Full System Prompt",
                    content=system_prompt[:500] + "...",
                    purpose="Complete system instructions",
                    testable_behaviors=["Overall instruction following"],
                    evaluation_criteria=[
                        EvaluationCriteria(
                            name="Overall Adherence",
                            criteria="How well does the response follow all system prompt instructions?",
                            threshold=0.7,
                        )
                    ],
                    potential_confusion_points=[f"Parse error: {str(e)}"],
                    latency_impact="medium",
                    complexity_score=0.5,
                )
            ]

    async def gather_optimization_goals_interactive(self) -> OptimizationGoals:
        """
        Interactive method to gather user's optimization goals.
        Returns default goals - actual interaction happens at CLI level.
        """
        # This would be called from CLI with actual user input
        # For now, return balanced defaults
        return OptimizationGoals()

    def get_all_evaluation_criteria(
        self,
        sections: List[SystemPromptSection],
    ) -> List[EvaluationCriteria]:
        """Extract all evaluation criteria from analyzed sections."""
        all_criteria = []
        for section in sections:
            all_criteria.extend(section.evaluation_criteria)
        return all_criteria

    async def generate_section_diagnostics(
        self,
        sections: List[SystemPromptSection],
        test_summary: TestRunSummary,
        goals: OptimizationGoals,
    ) -> List[SectionDiagnostic]:
        """
        Analyze test results against each section to generate diagnostics.

        Args:
            sections: Analyzed system prompt sections
            test_summary: Results from test execution
            goals: User's optimization goals

        Returns:
            List of SectionDiagnostic objects
        """
        # Collect all response data for analysis
        responses_data = []
        for result in test_summary.results:
            for response in result.responses:
                responses_data.append({
                    "model": response.model_id,
                    "prompt": result.prompt_content,
                    "response": response.content[:1000] if response.content else "",
                    "latency_ms": response.latency_ms,
                    "llm_latency_ms": response.llm_latency_ms,
                    "tool_latency_ms": response.tool_latency_ms,
                    "tool_calls": response.num_tool_calls,
                    "tools_used": response.tools_used,
                    "iterations": response.num_iterations,
                    "tokens": response.input_tokens + response.output_tokens,
                    "eval_scores": response.eval_scores or {},
                    "eval_reasons": response.eval_reasons or {},
                })

        # Build analysis prompt
        analysis_prompt = """You are an expert at diagnosing AI agent behavior against system prompt instructions.

Analyze the test results to determine how well the agent followed each section of the system prompt.
Identify issues, confusion patterns, and latency impacts.

For each section, provide:
1. Adherence score (0-1)
2. Specific issues found (with evidence from responses)
3. Latency impact in ms (estimated from the data)
4. Confusion indicators (loops, wrong tools, off-voice, etc.)
5. Token overhead (extra tokens due to this section's complexity)
6. Specific recommendation to improve this section
7. Priority level (critical/high/medium/low)
8. Before/after example if recommending changes

Return as JSON array:
```json
[
  {
    "section_name": "Section Name",
    "adherence_score": 0.75,
    "issues_found": ["Issue 1 with evidence", "Issue 2"],
    "latency_impact_ms": 1500,
    "confusion_indicators": ["Agent looped 3 times", "Used wrong tool"],
    "token_overhead": 200,
    "recommendation": "Specific actionable recommendation",
    "priority": "high",
    "before_example": "Current instruction text",
    "after_example": "Suggested improved text"
  }
]
```"""

        sections_json = json.dumps([
            {
                "name": s.name,
                "purpose": s.purpose,
                "testable_behaviors": s.testable_behaviors,
                "potential_confusion_points": s.potential_confusion_points,
            }
            for s in sections
        ], indent=2)

        user_message = f"""SYSTEM PROMPT SECTIONS:
{sections_json}

USER OPTIMIZATION GOALS:
- Latency priority: {goals.latency_priority}
- Tool accuracy priority: {goals.tool_accuracy_priority}
- Response quality priority: {goals.response_quality_priority}
- Cost priority: {goals.cost_priority}
- Voice adherence priority: {goals.voice_adherence_priority}
{f"- Custom goals: {', '.join(goals.custom_goals)}" if goals.custom_goals else ""}

TEST RESULTS:
{json.dumps(responses_data, indent=2)}

Analyze each section and return diagnostics as JSON array."""

        response = await self._call_llm(analysis_prompt, user_message)

        # Parse diagnostics
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]

            diagnostics_data = json.loads(json_str)

            diagnostics = []
            for d in diagnostics_data:
                diagnostic = SectionDiagnostic(
                    section_name=d["section_name"],
                    adherence_score=d.get("adherence_score", 0.5),
                    issues_found=d.get("issues_found", []),
                    latency_impact_ms=d.get("latency_impact_ms", 0),
                    confusion_indicators=d.get("confusion_indicators", []),
                    token_overhead=d.get("token_overhead", 0),
                    recommendation=d.get("recommendation", ""),
                    priority=d.get("priority", "medium"),
                    before_example=d.get("before_example"),
                    after_example=d.get("after_example"),
                )
                diagnostics.append(diagnostic)

            return diagnostics

        except (json.JSONDecodeError, KeyError) as e:
            return [
                SectionDiagnostic(
                    section_name="Analysis Error",
                    adherence_score=0.0,
                    issues_found=[f"Failed to parse diagnostics: {str(e)}"],
                    latency_impact_ms=0,
                    confusion_indicators=[],
                    token_overhead=0,
                    recommendation="Re-run analysis",
                    priority="critical",
                )
            ]

    async def generate_diagnostic_report(
        self,
        system_prompt: str,
        test_summary: TestRunSummary,
        goals: OptimizationGoals,
    ) -> DiagnosticReport:
        """
        Generate a complete diagnostic report for a system prompt.

        Args:
            system_prompt: The system prompt being analyzed
            test_summary: Results from test execution
            goals: User's optimization goals

        Returns:
            Complete DiagnosticReport
        """
        # Step 1: Decompose system prompt
        sections = await self.decompose_system_prompt(system_prompt)

        # Step 2: Generate section diagnostics
        diagnostics = await self.generate_section_diagnostics(
            sections, test_summary, goals
        )

        # Step 3: Calculate overall scores
        overall_scores = {
            "instruction_following": 0.0,
            "tool_accuracy": 0.0,
            "response_quality": 0.0,
            "latency_efficiency": 0.0,
            "cost_efficiency": 0.0,
        }

        # Aggregate from test results
        all_scores = []
        total_latency = 0
        total_cost = 0
        response_count = 0

        for result in test_summary.results:
            for response in result.responses:
                if response.eval_scores:
                    all_scores.append(response.eval_scores)
                total_latency += response.latency_ms
                total_cost += response.cost_usd
                response_count += 1

        if all_scores:
            # Average each score type
            score_sums = {}
            for scores in all_scores:
                for name, score in scores.items():
                    if name not in score_sums:
                        score_sums[name] = []
                    score_sums[name].append(score)

            for name, values in score_sums.items():
                key = name.lower().replace(" ", "_")
                if key in overall_scores:
                    overall_scores[key] = sum(values) / len(values)

        # Latency and cost efficiency (normalized)
        avg_latency = total_latency / response_count if response_count else 0
        avg_cost = total_cost / response_count if response_count else 0
        overall_scores["latency_efficiency"] = max(0, 1 - (avg_latency / 30000))  # 30s max
        overall_scores["cost_efficiency"] = max(0, 1 - (avg_cost / 0.05))  # $0.05 max

        # Step 4: Calculate goal achievement
        weights = goals.to_weights_dict()
        goal_achievement = {}
        for goal, weight in weights.items():
            if weight > 0:
                # Map goals to scores
                score_key = {
                    "latency": "latency_efficiency",
                    "tool_accuracy": "tool_accuracy",
                    "response_quality": "response_quality",
                    "cost": "cost_efficiency",
                    "voice_adherence": "instruction_following",
                }.get(goal, goal)
                goal_achievement[goal] = overall_scores.get(score_key, 0.5)

        # Step 5: Generate priority recommendations
        priority_recs = []
        for diag in sorted(diagnostics, key=lambda d: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(d.priority, 4)):
            if diag.recommendation and diag.priority in ["critical", "high"]:
                priority_recs.append(f"[{diag.priority.upper()}] {diag.section_name}: {diag.recommendation}")

        # Step 6: Estimate improvements
        estimated_improvement = {}
        total_latency_impact = sum(d.latency_impact_ms for d in diagnostics)
        if total_latency_impact > 0:
            estimated_improvement["latency"] = f"-{int(total_latency_impact)}ms potential"

        critical_count = len([d for d in diagnostics if d.priority == "critical"])
        if critical_count > 0:
            estimated_improvement["quality"] = f"+{critical_count * 10}% if critical issues fixed"

        # Create report
        import hashlib
        prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:8]

        return DiagnosticReport(
            system_prompt_hash=prompt_hash,
            analyzed_at=datetime.now(),
            optimization_goals=goals,
            sections=sections,
            section_diagnostics=diagnostics,
            overall_scores=overall_scores,
            goal_achievement=goal_achievement,
            priority_recommendations=priority_recs[:10],  # Top 10
            estimated_improvement=estimated_improvement,
        )

    def format_diagnostic_report(self, report: DiagnosticReport) -> str:
        """Format a diagnostic report as readable text."""
        lines = [
            "=" * 70,
            "SYSTEM PROMPT DIAGNOSTIC REPORT",
            "=" * 70,
            f"Prompt Hash: {report.system_prompt_hash}",
            f"Analyzed: {report.analyzed_at.isoformat()}",
            "",
            "OPTIMIZATION GOALS:",
            f"  Latency: {report.optimization_goals.latency_priority:.0%}",
            f"  Tool Accuracy: {report.optimization_goals.tool_accuracy_priority:.0%}",
            f"  Response Quality: {report.optimization_goals.response_quality_priority:.0%}",
            f"  Cost: {report.optimization_goals.cost_priority:.0%}",
            f"  Voice Adherence: {report.optimization_goals.voice_adherence_priority:.0%}",
            "",
            "-" * 70,
            "OVERALL SCORES",
            "-" * 70,
        ]

        for name, score in report.overall_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"  {name:25} [{bar}] {score:.2f}")

        lines.extend([
            "",
            "-" * 70,
            "GOAL ACHIEVEMENT",
            "-" * 70,
        ])

        for goal, achievement in report.goal_achievement.items():
            status = "✓" if achievement >= 0.7 else "⚠" if achievement >= 0.5 else "✗"
            lines.append(f"  {status} {goal:20} {achievement:.0%}")

        lines.extend([
            "",
            "-" * 70,
            "SECTION-BY-SECTION DIAGNOSTICS",
            "-" * 70,
        ])

        for diag in report.section_diagnostics:
            priority_icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(diag.priority, "⚪")
            lines.extend([
                "",
                f"{priority_icon} {diag.section_name} (Adherence: {diag.adherence_score:.0%})",
                f"   Priority: {diag.priority.upper()}",
            ])

            if diag.issues_found:
                lines.append("   Issues:")
                for issue in diag.issues_found[:3]:
                    lines.append(f"     • {issue[:80]}...")

            if diag.confusion_indicators:
                lines.append("   Confusion Indicators:")
                for indicator in diag.confusion_indicators[:3]:
                    lines.append(f"     ⚡ {indicator}")

            if diag.latency_impact_ms > 0:
                lines.append(f"   Latency Impact: +{diag.latency_impact_ms:.0f}ms")

            if diag.recommendation:
                lines.append(f"   Recommendation: {diag.recommendation[:100]}...")

            if diag.before_example and diag.after_example:
                lines.extend([
                    "   Before:",
                    f"     \"{diag.before_example[:60]}...\"",
                    "   After:",
                    f"     \"{diag.after_example[:60]}...\"",
                ])

        lines.extend([
            "",
            "-" * 70,
            "PRIORITY RECOMMENDATIONS",
            "-" * 70,
        ])

        for i, rec in enumerate(report.priority_recommendations, 1):
            lines.append(f"  {i}. {rec}")

        if report.estimated_improvement:
            lines.extend([
                "",
                "-" * 70,
                "ESTIMATED IMPROVEMENT POTENTIAL",
                "-" * 70,
            ])
            for metric, improvement in report.estimated_improvement.items():
                lines.append(f"  {metric}: {improvement}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)
