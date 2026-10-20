"""infra-assistant — a minimal Telegram assistant for infrastructure Q&A.

One LLM, a system prompt assembled from plain files, read-only tools from one or
more MCP servers, and an ordered provider fallback chain. The deliberate
anti-framework: no skills, no plugins, no autonomy, no scheduler. If the model
wants an action taken, it returns the command for a human to run.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
