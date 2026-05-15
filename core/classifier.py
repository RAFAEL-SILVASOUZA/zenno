"""Rule-based domain/complexity classifier.

The goal is to avoid the ~3s LLM classification cost for the majority of requests
that are obvious from surface features (greetings, factual lookups, code blocks,
math expressions, planning verbs). The LLM classifier is kept as a fallback for
the ambiguous middle.

Domains returned:
  conversation — greetings, small talk, acknowledgements
  factual      — who/what/when/where one-liners
  math         — arithmetic, probability, proofs, equations
  code         — programming tasks, algorithms, debugging
  logic        — logic puzzles, counterintuitive problems
  plan         — planning, trade-offs, comparisons
  unknown      — defer to LLM
"""
from __future__ import annotations

import re

# Each pattern list is OR'd; matching count contributes to confidence.

CONVERSATION_PATTERNS = [
    r"^\s*(hi|hello|hey|oi|olá|ola|bom dia|boa tarde|boa noite)\b",
    r"^\s*(thanks|thank you|obrigad[oa]|valeu|ok|okay|sure|got it)\s*[.!?]*\s*$",
    r"^\s*(yes|no|sim|n[aã]o)\s*[.!?]*\s*$",
    r"\bhow are you\b|\bcomo voc[eê] est[aá]\b|\btudo bem\b",
]

FACTUAL_PATTERNS = [
    r"^\s*(who|what|when|where|which)\s+(is|are|was|were|did|does|do)\b",
    r"^\s*(quem|o que|qual|quando|onde)\s+(é|foi|era|são|eram)\b",
    r"\bcapital of\b|\bcapital d[aeo]\b",
    r"\bdefinition of\b|\bdefini[cç][aã]o de\b",
    r"\btranslate\b|\btraduz[ai]r?\b",
]

MATH_PATTERNS = [
    r"\bprobab(?:ility|ilidade)\b",
    r"\b(expected value|valor esperado)\b",
    r"\b(combinator|permutat|factorial|fatorial)\w*",
    r"\b(integral|derivative|derivada|integral)\b",
    r"\b(equation|equa[cç][aã]o|solve for|resolva)\b",
    r"\b(prove|theorem|lemma|proof|prova|teorema|demonstre)\b",
    r"\bsqrt\b|\bsum_|\bprod_",
    r"\b\d+\s*[+\-*/^%]\s*\d+",                # arithmetic expression
    r"\b(modul[oa]|mod)\s+\d+",
    r"\b(simplify|fatori[sz]e|factori[sz]e)\b",
    r"\b(matrix|matriz|vetor|vector|linear algebra|álgebra linear)\b",
]

CODE_PATTERNS = [
    r"```[a-z]*\s*\n",                          # fenced code block in prompt
    r"\bdef\s+\w+\s*\(",                        # python def
    r"\bfunction\s+\w+\s*\(",                   # JS function
    r"\b(implement|refactor|debug)\b",
    r"\bfix\s+(?:this|the|my)\s+(?:code|bug|function|script)\b",
    r"\bwrite\s+(?:a|an|me|some|that|this)(?:\s+\w+)?\s+(?:function|class|method|script|program|code)\b",
    r"\bescreva\s+(?:uma|um|esse|este)(?:\s+\w+)?\s+(?:fun[cç][aã]o|classe|m[eé]todo|script|programa|c[oó]digo)\b",
    r"\b(algorithm|algoritmo)\b",
    r"\b(complexity|complexidade|big-?o)\b",
    r"\b(python|javascript|typescript|java|rust|golang|c\+\+|cpp|kotlin|swift)\b",
    r"\b(api|endpoint|database|sql|query|regex|regexp)\b",
    r"\b(class|interface|struct)\s+[A-Z]\w*",
]

LOGIC_PATTERNS = [
    r"\b(if and only if|iff)\b",
    r"\b(truth table|tabela verdade)\b",
    r"\b(tautolog|contradiction|contradi[cç][aã]o)\w*",
    r"\bknights?\s+and\s+knaves?\b",
    r"\bcounter-?intuitive\b",
    r"\b(monty hall|paradox|paradoxo)\b",
]

PLAN_PATTERNS = [
    r"\b(plan|plano|planeje|roadmap|itinerary|itiner[aá]rio)\b",
    r"\b(strategy|estrat[eé]gia|trade-?off|compar[ae])\b",
    r"\b(pros and cons|pr[oó]s e contras|step-?by-?step)\b",
    r"\b(how (do|to|would|should) i)\b",
    r"\bbest way to\b|\bmelhor (jeito|forma) (de|para)\b",
]


def _count_hits(text: str, patterns: list[str]) -> int:
    return sum(1 for p in patterns if re.search(p, text, flags=re.IGNORECASE))


def classify(text: str) -> tuple[str, str, float]:
    """Classify a user message.

    Returns (domain, strategy, confidence) where:
      strategy ∈ {"direct", "reasoning", "unknown"}
      confidence ∈ [0.0, 1.0]; <0.7 means caller should fall back to LLM.
    """
    if not text:
        return "conversation", "direct", 0.99

    t = text.strip()

    # very short messages → conversation
    if len(t) < 25:
        if _count_hits(t, CONVERSATION_PATTERNS):
            return "conversation", "direct", 0.95
        # short but not greeting — could be factual one-liner
        if _count_hits(t, FACTUAL_PATTERNS):
            return "factual", "direct", 0.85

    # Strong signals first
    code_hits  = _count_hits(t, CODE_PATTERNS)
    math_hits  = _count_hits(t, MATH_PATTERNS)
    logic_hits = _count_hits(t, LOGIC_PATTERNS)
    plan_hits  = _count_hits(t, PLAN_PATTERNS)
    fact_hits  = _count_hits(t, FACTUAL_PATTERNS)
    chat_hits  = _count_hits(t, CONVERSATION_PATTERNS)

    # Code wins over generic factual when a code block is present
    if code_hits >= 2 or (code_hits >= 1 and "```" in t):
        return "code", "reasoning", min(0.95, 0.7 + 0.1 * code_hits)

    # Single strong code hit + explicit action verb is still confident enough.
    if code_hits >= 1 and re.search(
        r"\b(write|implement|debug|fix|refactor|create|build|generate"
        r"|escreva|implemente|crie|gere|corrija)\b",
        t, flags=re.IGNORECASE,
    ):
        return "code", "reasoning", 0.78

    # Math signals are usually unambiguous
    if math_hits >= 2 or (math_hits >= 1 and logic_hits + code_hits == 0):
        return "math", "reasoning", min(0.95, 0.7 + 0.1 * math_hits)

    if logic_hits >= 1:
        return "logic", "reasoning", min(0.9, 0.7 + 0.1 * logic_hits)

    if plan_hits >= 2:
        return "plan", "reasoning", min(0.9, 0.65 + 0.1 * plan_hits)

    if fact_hits >= 1 and code_hits == 0 and math_hits == 0:
        return "factual", "direct", min(0.9, 0.7 + 0.05 * fact_hits)

    if chat_hits >= 1 and len(t) < 80:
        return "conversation", "direct", 0.8

    # ambiguous — let the LLM decide
    return "unknown", "unknown", 0.0
