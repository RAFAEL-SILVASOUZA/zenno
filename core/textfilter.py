"""Strip internal-reasoning blocks from model output.

Many reasoning-tuned models (DeepSeek R1, gpt-oss, QwQ, Qwen3-thinking, …)
emit a chain of thought before the final answer. When Zenno is used as the
backend for an agent (Claude Code, Open WebUI, custom tool-using agents),
that prefix breaks the agent: it gets fed back as the assistant's message.

We support two operating modes:

  - Post-hoc `strip(text)` for the non-streaming path: applies regex passes.
  - `StreamingFilter` for the streaming path: an incremental state machine
    that feeds chunks in and emits sanitized chunks out, holding a small
    look-behind so partial tag prefixes never leak.

Markers handled
---------------
XML-style (any of):
    <think>...</think>
    <thinking>...</thinking>
    <thought>...</thought>
    <reasoning>...</reasoning>
    <analysis>...</analysis>

gpt-oss / harmony tokens:
    <|channel|>analysis<|message|> ... <|end|>          -> drop entirely
    <|channel|>final<|message|>    ... <|return|>       -> in `strip()`, ONLY
                                                          this content is kept
    Stray <|start|> / <|end|> / <|return|> / <|message|> / <|channel|> tokens -> stripped
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_XML_TAGS = ("think", "thinking", "thought", "reasoning", "analysis")
_OPEN_MARKERS = tuple(f"<{t}>" for t in _XML_TAGS)
_CLOSE_MARKERS = tuple(f"</{t}>" for t in _XML_TAGS)

_HARMONY_OPEN_ANALYSIS = "<|channel|>analysis<|message|>"
_HARMONY_OPEN_FINAL    = "<|channel|>final<|message|>"
_HARMONY_END_TOKENS    = ("<|end|>", "<|return|>")

# Post-hoc regexes
_XML_BLOCK = re.compile(
    r"<\s*(" + "|".join(_XML_TAGS) + r")\s*>.*?</\s*\1\s*>",
    flags=re.DOTALL | re.IGNORECASE,
)
_HARMONY_ANALYSIS = re.compile(
    r"<\|channel\|>\s*analysis\s*<\|message\|>.*?(?:<\|end\|>|<\|return\|>)",
    flags=re.DOTALL,
)
_HARMONY_FINAL = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|\Z)",
    flags=re.DOTALL,
)
_HARMONY_TOKENS = re.compile(
    r"<\|(?:start|end|return|message|channel|im_start|im_end)\|>"
    r"(?:[a-zA-Z]+<\|message\|>)?"
)
# Catches stray "<think>" or "</think>" left over by malformed streams.
_STRAY_TAG = re.compile(
    r"</?\s*(?:" + "|".join(_XML_TAGS) + r")\s*>",
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Post-hoc strip
# ---------------------------------------------------------------------------

_UNCLOSED_XML = re.compile(
    r"<\s*(?:" + "|".join(_XML_TAGS) + r")\s*>.*",
    flags=re.DOTALL | re.IGNORECASE,
)
_UNCLOSED_HARMONY = re.compile(
    r"<\|channel\|>\s*analysis\s*<\|message\|>.*",
    flags=re.DOTALL,
)


def strip(text: str) -> str:
    """Strip internal-reasoning markers from a complete text.

    If the text contains a harmony `final` channel, ONLY the content of that
    channel is returned — that is the model's intended user-facing answer.
    """
    if not text:
        return text

    # 1) Harmony final channel wins outright: it is by definition the answer.
    m = _HARMONY_FINAL.search(text)
    if m:
        candidate = m.group(1)
        candidate = _STRAY_TAG.sub("", candidate)
        candidate = _HARMONY_TOKENS.sub("", candidate)
        return candidate.strip()

    # 2) Drop XML-tagged blocks and harmony analysis blocks.
    clean = _HARMONY_ANALYSIS.sub("", text)
    clean = _XML_BLOCK.sub("", clean)

    # 3) Drop any UNCLOSED reasoning block (open tag with no matching close).
    #    A reasoning model that crashed mid-stream still leaks safer this way:
    #    we lose its (already incomplete) answer instead of leaking the trace.
    clean = _UNCLOSED_XML.sub("", clean)
    clean = _UNCLOSED_HARMONY.sub("", clean)

    # 4) Mop up any stray markers left by malformed output.
    clean = _STRAY_TAG.sub("", clean)
    clean = _HARMONY_TOKENS.sub("", clean)

    return clean.strip()


# ---------------------------------------------------------------------------
# Streaming filter
# ---------------------------------------------------------------------------

class StreamingFilter:
    """Incremental thinking-block stripper for streamed text.

    Usage::

        f = StreamingFilter()
        for piece in stream:
            clean = f.feed(piece)
            if clean:
                emit(clean)
        tail = f.flush()
        if tail:
            emit(tail)

    Guarantees:
      - Never emits any character that is inside a known reasoning block.
      - Holds back a small look-behind window so partial markers (e.g. "<thi")
        are never flushed prematurely.
      - Safe to call `feed("")` and `flush()` multiple times.
    """

    _MAX_LOOKBEHIND = max(
        max(len(t) for t in _OPEN_MARKERS),
        max(len(t) for t in _CLOSE_MARKERS),
        len(_HARMONY_OPEN_ANALYSIS),
        len(_HARMONY_OPEN_FINAL),
        max(len(t) for t in _HARMONY_END_TOKENS),
    )

    def __init__(self) -> None:
        self._buf = ""
        self._in_block = False

    # ---- public --------------------------------------------------------------

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._buf += text
        return self._drain()

    def flush(self) -> str:
        """Drain everything safe to emit. Any unterminated block is discarded."""
        if self._in_block:
            self._buf = ""
            self._in_block = False
            return ""
        # If the buffer ends in a partial marker (e.g. '<thi'), drop the
        # partial prefix — we cannot know if a closing was coming.
        boundary = self._safe_emit_boundary()
        out = self._buf[:boundary]
        self._buf = ""
        out = _STRAY_TAG.sub("", out)
        out = _HARMONY_TOKENS.sub("", out)
        return out

    # ---- internals -----------------------------------------------------------

    def _drain(self) -> str:
        out_parts: list[str] = []
        while self._buf:
            if self._in_block:
                close_idx, close_len = self._find_close()
                if close_idx < 0:
                    # Still inside an analysis block; hold buffer.
                    return "".join(out_parts)
                # Drop the block content and the close marker.
                self._buf = self._buf[close_idx + close_len:]
                self._in_block = False
                continue

            open_idx, open_len = self._find_open()
            if open_idx < 0:
                safe_end = self._safe_emit_boundary()
                if safe_end <= 0:
                    return "".join(out_parts)
                chunk = self._buf[:safe_end]
                self._buf = self._buf[safe_end:]
                # Strip any complete harmony noise tokens before emitting.
                chunk = _HARMONY_TOKENS.sub("", chunk)
                chunk = _STRAY_TAG.sub("", chunk)
                out_parts.append(chunk)
                return "".join(out_parts)

            # Found an opening marker: emit text before it, drop the marker,
            # flip state.
            prefix = self._buf[:open_idx]
            self._buf = self._buf[open_idx + open_len:]
            prefix = _HARMONY_TOKENS.sub("", prefix)
            prefix = _STRAY_TAG.sub("", prefix)
            out_parts.append(prefix)
            self._in_block = True

        return "".join(out_parts)

    def _find_open(self) -> tuple[int, int]:
        """First open marker (index, length) or (-1, 0)."""
        lowered = self._buf.lower()
        best_idx = -1
        best_len = 0
        for tag in _OPEN_MARKERS:
            idx = lowered.find(tag)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx, best_len = idx, len(tag)
        # Harmony analysis open
        idx = lowered.find(_HARMONY_OPEN_ANALYSIS)
        if idx >= 0 and (best_idx < 0 or idx < best_idx):
            best_idx, best_len = idx, len(_HARMONY_OPEN_ANALYSIS)
        return best_idx, best_len

    def _find_close(self) -> tuple[int, int]:
        lowered = self._buf.lower()
        best_idx = -1
        best_len = 0
        for tag in _CLOSE_MARKERS:
            idx = lowered.find(tag)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx, best_len = idx, len(tag)
        for tok in _HARMONY_END_TOKENS:
            idx = lowered.find(tok)
            if idx >= 0 and (best_idx < 0 or idx < best_idx):
                best_idx, best_len = idx, len(tok)
        return best_idx, best_len

    _ALL_MARKERS = (
        _OPEN_MARKERS
        + _CLOSE_MARKERS
        + (_HARMONY_OPEN_ANALYSIS, _HARMONY_OPEN_FINAL)
        + _HARMONY_END_TOKENS
    )

    def _safe_emit_boundary(self) -> int:
        """Return index up to which the buffer can be emitted safely.

        We must never emit a trailing partial marker. For each known marker we
        check whether the buffer ENDS with any non-empty prefix of it; if so,
        we hold from where that prefix begins. The deepest hold wins.
        """
        n = len(self._buf)
        if n == 0:
            return 0
        lowered = self._buf.lower()
        cutoff = n
        for marker in self._ALL_MARKERS:
            mlow = marker.lower()
            # Longest prefix of marker that buf ends with; check long → short.
            limit = min(len(mlow), n)
            for k in range(limit, 0, -1):
                if lowered.endswith(mlow[:k]):
                    start = n - k
                    if start < cutoff:
                        cutoff = start
                    break
        return cutoff
