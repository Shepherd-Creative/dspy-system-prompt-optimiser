# RAG System Message Optimiser Agent

A tool that helps you **improve the instructions you give to AI assistants** (like ChatGPT, Claude, or Gemini).

## What Problem Does This Solve?

When you build an AI-powered application, you write a "system prompt" - the hidden instructions that tell the AI how to behave. But how do you know if your instructions are actually working?

This tool answers: **What parts of my AI instructions are working, what's broken, and how do I fix them?**

## How It Works

1. **Analyse** - The system breaks down your instructions into testable sections and identifies potential problem areas

2. **Test** - It runs your instructions through multiple AI models (Claude, Gemini, GPT-4, etc.) with real user scenarios

3. **Evaluate** - An AI judge scores how well each model followed your instructions, checking:
   - Did it use the right tools?
   - Did it respond in the right style/voice?
   - Was the response helpful and accurate?

4. **Report** - You get a diagnostic report showing which parts of your instructions work well and which need improvement, with specific recommendations

## Architecture

The system uses a 3-layer architecture for reliability:

```
┌─────────────────────────────────────────────────┐
│  Directives    │  What to do (Markdown SOPs)    │
├─────────────────────────────────────────────────┤
│  Orchestration │  AI decision-making layer      │
├─────────────────────────────────────────────────┤
│  Execution     │  Deterministic Python scripts  │
└─────────────────────────────────────────────────┘
```

This separation ensures that complex operations are handled by tested, reliable code rather than unpredictable AI generation.

## Quick Start

1. Clone the repository
2. Copy `.env.example` to `.env` and add your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run a diagnostic: `python scripts/run_diagnostic.py`

## Requirements

- Python 3.11+
- API key for [OpenRouter](https://openrouter.ai) (provides access to multiple AI models)

## Project Structure

```
├── directives/          # Instructions for the AI orchestrator
├── execution/           # Python scripts that do the actual work
├── config/              # Model and tool configurations
├── scripts/             # Entry points for running diagnostics
└── tests/               # Test suite
```

## License

MIT
