"""Python sandbox for verifying model output.

This module exists because a model judging its own work has the same blind spots
as the model that wrote the work. For numerical and code answers we can do far
better: run the code, see what actually happens, and feed the real output back
to the evaluator. That is a *ground-truth signal* the LLM cannot fabricate.

Safety model:
  - Runs in an isolated Python subprocess with `-I` (no user site, ignore env).
  - Hard wall-clock timeout via subprocess.run(timeout=...).
  - A static-pattern denylist rejects code that touches the filesystem, network,
    or process control before launch.
  - Output is truncated to keep the evaluator prompt small.

Limits: this is NOT a hardened sandbox. It is "good enough" for verifying small
self-contained snippets emitted by your local Ollama. Do not expose this proxy
to untrusted user input without an OS-level sandbox (firejail, gVisor, etc).
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys

log = logging.getLogger("zenno")


# Code block extractors — prefer language-tagged blocks but fall back to bare.
_TAGGED_BLOCK = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BARE_BLOCK   = re.compile(r"```\s*\n(.*?)```", re.DOTALL)

# Static denylist applied BEFORE execution. The goal is to make the sandbox
# obviously safe for arithmetic/math/algorithm verification, not to be a general
# Python jail. Anything that looks like file/net/subprocess access is rejected.
_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bos\s*\.\s*(remove|rmdir|unlink|kill|system|popen|exec|spawn)", "os mutation"),
    (r"\bshutil\b",                                                     "shutil"),
    (r"\bsubprocess\b",                                                 "subprocess"),
    (r"\b__import__\s*\(\s*['\"](?:os|sys|subprocess|socket|ctypes)",  "dynamic dangerous import"),
    (r"\bsocket\b",                                                     "socket"),
    (r"\bctypes\b",                                                     "ctypes"),
    (r"\bopen\s*\([^)]*['\"][wax]",                                     "file write"),
    (r"\brequests\b",                                                   "requests"),
    (r"\burllib\b",                                                     "urllib"),
    (r"\bhttplib?\b",                                                   "http"),
    (r"\bexec\s*\(",                                                    "exec()"),
]

# Extract a final-answer line — used for self-consistency voting on math problems.
_FINAL_LINE = re.compile(
    r"(?:final\s+answer|resposta\s+final|resposta|answer)\s*[:=]\s*([^\n]+)",
    re.IGNORECASE,
)
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?(?:/\d+)?")


def extract_python(text: str) -> str | None:
    """Pull the first Python-looking code block from *text*. Returns None if none."""
    if not text:
        return None
    m = _TAGGED_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    m = _BARE_BLOCK.search(text)
    if m:
        code = m.group(1).strip()
        # Heuristic: only treat as Python if it has Python keywords.
        if re.search(r"\b(def|import|print|return|for|while|if|class)\b", code):
            return code
    return None


def is_safe(code: str) -> tuple[bool, str]:
    """Static check before running. Returns (safe, reason_if_not)."""
    for pat, name in _FORBIDDEN_PATTERNS:
        if re.search(pat, code):
            return False, name
    return True, ""


def run_python(code: str, timeout: float = 5.0) -> dict:
    """Execute *code* in a sandboxed Python subprocess.

    Returns a dict:
      ok          — exit code was 0
      stdout      — captured stdout (tail-truncated)
      stderr      — captured stderr (tail-truncated)
      returncode  — process exit code, or negative for sandbox failures
      skipped     — True if denylist rejected the code (then ok=False)
      reason      — human-readable reason if skipped
    """
    safe, reason = is_safe(code)
    if not safe:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"SANDBOX REJECTED: forbidden ({reason})",
            "returncode": -10,
            "skipped": True,
            "reason": reason,
        }

    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],   # -I: isolated mode
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
            "returncode": proc.returncode,
            "skipped": False,
            "reason": "",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"TIMEOUT after {timeout}s",
            "returncode": -20,
            "skipped": False,
            "reason": "timeout",
        }
    except Exception as exc:
        log.warning("sandbox exception: %s", exc)
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"SANDBOX ERROR: {exc}",
            "returncode": -30,
            "skipped": False,
            "reason": "exception",
        }


def extract_final_answer(text: str) -> str | None:
    """Try to extract the canonical final answer from a reasoning trace.

    Used by self-consistency voting: same answer extracted from N samples means
    high agreement.
    """
    if not text:
        return None
    matches = _FINAL_LINE.findall(text)
    if matches:
        last = matches[-1].strip().rstrip(".!?;,")
        nums = _NUMBER.findall(last)
        if nums:
            return nums[-1].replace(",", ".")
        return last[:120]
    nums = _NUMBER.findall(text)
    if nums:
        return nums[-1].replace(",", ".")
    return None
