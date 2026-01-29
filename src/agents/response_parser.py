# src/agents/response_parser.py
"""
LLM response parsing utilities.

This module handles parsing LLM responses into structured decisions,
handling various response formats (JSON in code blocks, raw JSON, etc.).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .base import Order


def parse_json_response(content: str) -> Dict[str, Any]:
    """
    Parse a simple JSON response from LLM output.

    Handles various formats:
    - Raw JSON
    - JSON in ```json code blocks
    - JSON in ``` code blocks
    - JSON embedded in other text

    Args:
        content: Raw LLM response text

    Returns:
        Parsed JSON as dict, or default hold action on failure
    """
    text = content.strip()

    # Extract from code blocks if present
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find JSON object (simple, non-nested)
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        text = match.group()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"action": "hold", "price": 0, "quantity": 0}


def parse_structured_response(content: str) -> Dict[str, Any]:
    """
    Parse a structured CoT response with nested decision.

    Expected format:
    {
        "analysis": "...",
        "reasoning": "...",
        "decision": {"action": "buy", "price": 10, "quantity": 1}
    }

    Args:
        content: Raw LLM response text

    Returns:
        Parsed JSON as dict with analysis, reasoning, and decision
    """
    text = content.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    # Find the outermost JSON object (allows nested braces)
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        text = match.group()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "analysis": "Failed to parse",
            "reasoning": content[:200],
            "decision": {"action": "hold", "price": 0, "quantity": 0}
        }


def decision_to_order(decision: Dict[str, Any], agent_id: str) -> Optional[Order]:
    """
    Convert a decision dict to an Order object.

    Args:
        decision: Dict with action, price, quantity keys
        agent_id: ID of the agent making the order

    Returns:
        Order object, or None if action is "hold" or invalid
    """
    action = decision.get("action", "hold")
    if action not in ["buy", "sell"]:
        return None

    try:
        return Order(
            agent_id=agent_id,
            side=action,
            good="good_A",
            price=float(decision.get("price", 0)),
            quantity=float(decision.get("quantity", 1))
        )
    except (ValueError, TypeError):
        return None
