import asyncio
import logging
import random
import time
import uuid
from collections import Counter

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from core.classifier import classify as heuristic_classify
from core.config import settings
from core.domains import get_config as get_domain_config
from core.sandbox import extract_final_answer, extract_python, run_python
from core.textfilter import StreamingFilter, strip as strip_thinking
from graph.state import GraphState

log = logging.getLogger("zenno")

# Global registry: request_id -> asyncio.Queue
streaming_registry: dict[str, asyncio.Queue] = {}


def _text(content) -> str:
    """Extract plain text from a string or a multimodal content list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        timeout=settings.api_request_timeout,
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError)


async def _api_call(request_id: str, func, max_retries: int | None = None) -> any:
    """Execute an async API call with exponential backoff retry on transient errors."""
    retries = max_retries if max_retries is not None else settings.max_api_retries

    for attempt in range(1, retries + 1):
        try:
            return await func()
        except _RETRYABLE_ERRORS as exc:
            if attempt >= retries:
                log.error("[%s] API call failed after %d attempt(s): %s", request_id, attempt, exc)
                raise
            delay = min(settings.retry_base_delay * (2 ** (attempt - 1)), settings.retry_max_delay)
            jitter = delay * random.uniform(0.5, 1.0)
            log.warning("[%s] API transient error (attempt %d/%d): %s — retrying in %.1fs",
                        request_id, attempt, retries, exc, jitter)
            await asyncio.sleep(jitter)


async def _safe_api_call(request_id: str, func, fallback=None) -> any:
    """Like _api_call but returns *fallback* on unrecoverable failure instead of raising."""
    try:
        return await _api_call(request_id, func)
    except Exception as exc:
        log.error("[%s] API call exhausted all retries: %s", request_id, exc)
        return fallback


def _sse_chunk(content: str, model: str, response_id: str, finish: bool = False) -> dict:
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {} if finish else {"content": content},
                "finish_reason": "stop" if finish else None,
            }
        ],
    }


async def _push(request_id: str, chunk: dict) -> None:
    q = streaming_registry.get(request_id)
    if q:
        await q.put(chunk)


async def _close(request_id: str) -> None:
    q = streaming_registry.get(request_id)
    if q:
        await q.put(None)


def _last_user_text(state: GraphState) -> str:
    return _text(next(
        (m["content"] for m in reversed(state["messages"]) if m["role"] == "user"),
        "",
    ))


def _merge_system_prompt(messages: list[dict], extra_system: str) -> list[dict]:
    """Merge *extra_system* with any existing system messages.

    Some model templates (Gemma, etc.) require exactly one system message at
    index 0. We collapse all existing systems plus our domain prompt into a
    single leading message and keep the rest of the conversation intact.
    """
    if not extra_system:
        return list(messages)

    existing_systems = [m for m in messages if m.get("role") == "system"]
    non_system       = [m for m in messages if m.get("role") != "system"]

    parts: list[str] = []
    for m in existing_systems:
        c = _text(m.get("content", ""))
        if c:
            parts.append(c)
    parts.append(extra_system)
    merged = "\n\n".join(parts)

    return [{"role": "system", "content": merged}, *non_system]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def classify_node(state: GraphState) -> GraphState:
    """Decide complexity AND domain.

    Strategy:
      1) Try the rule-based heuristic classifier. It is free, returns a domain,
         and is unambiguous for greetings, factual one-liners, math expressions,
         code blocks, planning verbs, etc.
      2) If confidence is below `classify_heuristic_threshold`, fall back to the
         LLM classifier (the old behavior) and combine results.
    """
    rid = state["request_id"]
    last_user = _last_user_text(state)
    has_tools = bool(state.get("tools"))

    log.info("[%s] CLASSIFY | question: %.120s  tools=%s",
             rid, last_user, has_tools)

    # ---- 1) Heuristic pass ----------------------------------------------------
    h_domain, h_strategy, h_conf = heuristic_classify(last_user)
    log.info("[%s] CLASSIFY | heuristic domain=%s strategy=%s conf=%.2f",
             rid, h_domain, h_strategy, h_conf)

    # When tools are present, the agent should generally answer "simple" so the
    # caller can use them — the heuristic decision still gives us a useful
    # domain label for logging but the routing collapses to "simple".
    if has_tools and h_conf >= settings.classify_heuristic_threshold:
        log.info("[%s] CLASSIFY | tools present → route=SIMPLE (domain=%s)",
                 rid, h_domain)
        return {**state, "complexity": "simple", "domain": h_domain}

    if h_conf >= settings.classify_heuristic_threshold and h_strategy != "unknown":
        complexity = "complex" if h_strategy == "reasoning" else "simple"
        log.info("[%s] CLASSIFY | heuristic → route=%s domain=%s",
                 rid, complexity.upper(), h_domain)
        return {**state, "complexity": complexity, "domain": h_domain}

    # ---- 2) LLM fallback ------------------------------------------------------
    base_prompt = (
        'Classify the user request as "simple" or "complex". '
        "Answer with ONLY that one word.\n\n"
        "simple — answerable with a single fact or a one-shot reply:\n"
        "  • greetings, small talk\n"
        '  • single-fact lookups ("capital of France", "who wrote Hamlet")\n'
        "  • direct conversions, basic translation, short definitions\n"
        '  • short creative tasks ("write a haiku about rain")\n\n'
        "complex — benefits from step-by-step reasoning AND does NOT need external data:\n"
        "  • math/word problems, probability, combinatorics, logic puzzles\n"
        "  • problems where the obvious answer is often wrong (counterintuitive)\n"
        "  • pure code/algorithm analysis from text alone (no files to read)\n"
        "  • planning, comparison, trade-off analysis from given info\n"
        "  • anything requiring enumeration of cases or multi-step inference\n"
    )

    if has_tools:
        tool_note = (
            "\nIMPORTANT — tools are available to the assistant in this request.\n"
            "If the request requires looking up external information (files on disk,\n"
            "running commands, web searches, current data, listing/reading anything),\n"
            "answer SIMPLE — the assistant will use tools directly. Pure reasoning\n"
            "without external data → COMPLEX. When in doubt with tools present,\n"
            "prefer SIMPLE.\n"
        )
    else:
        tool_note = "\nWhen in doubt, answer complex.\n"

    examples = (
        "\nExamples:\n"
        "  Request: Hi, how are you?                                     -> simple\n"
        "  Request: Translate 'good morning' to Japanese                 -> simple\n"
        "  Request: What is the capital of Brazil?                       -> simple\n"
        "  Request: List the files in /tmp                               -> simple  (needs tool)\n"
        "  Request: Read main.py and tell me what it does                -> simple  (needs tool)\n"
        "  Request: Maria has two kids. At least one is a boy born on a  -> complex\n"
        "           Tuesday. What is the probability both are boys?\n"
        "  Request: Prove that sqrt(2) is irrational                     -> complex\n"
        "  Request: Plan a 5-day trip to Tokyo with a $2000 budget       -> complex\n\n"
        f"Request: {last_user}\n\nAnswer:"
    )

    prompt = base_prompt + tool_note + examples

    client = _client()
    response = await _api_call(rid, lambda: client.chat.completions.create(
        model=state["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ))

    raw = strip_thinking(response.choices[0].message.content or "").strip().lower()
    # Bias to the safe side: anything that is not explicitly "simple" → complex.
    if "complex" in raw:
        complexity = "complex"
    elif "simple" in raw:
        complexity = "simple"
    else:
        complexity = "complex"

    # If the heuristic had a guess (low confidence) keep it as the domain label;
    # otherwise pick a generic bucket so domain config still resolves to
    # something sane.
    if h_domain != "unknown":
        domain = h_domain
    else:
        domain = "complex" if complexity == "complex" else "simple"

    log.info("[%s] CLASSIFY | llm-fallback raw=%r → route=%s domain=%s",
             rid, raw, complexity.upper(), domain)
    return {**state, "complexity": complexity, "domain": domain}


async def direct_response_node(state: GraphState) -> GraphState:
    """Stream Ollama response directly for simple requests and all tool-call requests."""
    rid = state["request_id"]
    has_tools = bool(state.get("tools"))
    log.info("[%s] DIRECT | stream=%s tools=%s domain=%s",
             rid, state["do_stream"], has_tools, state.get("domain", "?"))
    client = _client()

    # For simple-but-classified requests (factual, conversation, short translation)
    # an explicit short system prompt sharpens the answer at near-zero cost. We
    # only do this when the caller did NOT pass tools — tool callers may have
    # carefully crafted system prompts we should not touch.
    messages = state["messages"]
    if not has_tools:
        cfg = get_domain_config(state.get("domain") or "simple")
        if cfg["system"]:
            messages = _merge_system_prompt(messages, cfg["system"])

    kwargs: dict = {
        "model": state["model"],
        "messages": messages,
        "temperature": state["temperature"],
    }
    if state.get("tools"):
        kwargs["tools"] = state["tools"]
    if state.get("tool_choice") is not None:
        kwargs["tool_choice"] = state["tool_choice"]

    if state["do_stream"]:
        # Filter <think>/<thinking>/harmony-analysis blocks out of streamed
        # content deltas. The filter holds a small look-behind so partial
        # markers never leak. Tool-call chunks and finish-reason chunks pass
        # through untouched.
        #
        # llama.cpp's OpenAI-compatible server (and LM Studio) emits the
        # reasoning trace in a separate `reasoning_content` delta field on
        # gpt-oss / DeepSeek-R1 style models. That field is non-standard and
        # downstream agents render it as part of the assistant message,
        # breaking tool-use loops. We drop the field entirely; if a chunk
        # carries nothing else of interest, we swallow the whole chunk.
        thinker = StreamingFilter()
        try:
            stream = await _api_call(rid, lambda: client.chat.completions.create(**kwargs, stream=True))
            async for chunk in stream:
                d = chunk.model_dump()
                choices = d.get("choices") or []
                if not choices:
                    await _push(rid, d)
                    continue
                delta = choices[0].get("delta") or {}

                # 1) Drop any reasoning_content field outright.
                had_reasoning = delta.pop("reasoning_content", None) is not None

                content = delta.get("content")
                if isinstance(content, str) and content:
                    safe = thinker.feed(content)
                    if not safe:
                        # Nothing safe to emit yet. If the chunk carries other
                        # useful metadata (tool_calls / finish_reason), forward
                        # it without the content field so the client still sees
                        # the signal.
                        if delta.get("tool_calls") or choices[0].get("finish_reason"):
                            delta.pop("content", None)
                            await _push(rid, d)
                        continue
                    delta["content"] = safe
                elif had_reasoning and not delta.get("tool_calls") \
                        and not choices[0].get("finish_reason") \
                        and not delta.get("role"):
                    # Chunk was reasoning-only — drop it entirely.
                    continue

                await _push(rid, d)

            # Drain any text held back by the filter (e.g. trailing buffered
            # chars after the last chunk). Push as a final content-only chunk.
            tail = thinker.flush()
            if tail:
                await _push(rid, _sse_chunk(tail, state["model"],
                                            f"chatcmpl-{uuid.uuid4().hex[:12]}"))

            await _close(rid)
            log.info("[%s] DIRECT | stream complete", rid)
        except Exception as exc:
            log.error("[%s] DIRECT | stream error: %s", rid, exc)
            await _push(rid, {"error": f"Streaming failed: {exc}"})
            await _close(rid)
        return {**state, "final_response": "", "tool_calls": None, "finish_reason": "stop"}
    else:
        response = await _safe_api_call(rid, lambda: client.chat.completions.create(**kwargs))
        if response is None:
            log.error("[%s] DIRECT | API completely failed, returning empty", rid)
            return {**state, "final_response": "", "tool_calls": None, "finish_reason": "stop"}

        choice = response.choices[0]
        tool_calls = None
        if choice.message.tool_calls:
            tool_calls = [tc.model_dump() for tc in choice.message.tool_calls]
            log.info("[%s] DIRECT | finish_reason=tool_calls  calls=%s",
                     rid, [tc["function"]["name"] for tc in tool_calls])
            final_text = ""
        else:
            # llama.cpp / LM Studio expose the reasoning trace on a separate
            # `reasoning_content` attribute. We deliberately ignore it — the
            # user-facing answer is `content` alone. strip_thinking handles the
            # case where the model decided to inline its trace into content.
            raw = choice.message.content or ""
            final_text = strip_thinking(raw)
            log.info("[%s] DIRECT | finish_reason=stop  chars=%d (stripped from %d)",
                     rid, len(final_text), len(raw))
        return {
            **state,
            "final_response": final_text,
            "tool_calls": tool_calls,
            "finish_reason": choice.finish_reason or "stop",
        }


# ---------------------------------------------------------------------------
# Self-consistency synthesis
# ---------------------------------------------------------------------------

def _synthesize(samples: list[str], domain: str) -> tuple[str, dict]:
    """Pick the best candidate among *samples* for a given *domain*.

    Returns (chosen_text, info) where info logs how the choice was made.
    """
    if len(samples) == 1:
        return samples[0], {"strategy": "single", "agreement": 1.0}

    if domain == "math":
        # Majority vote on extracted final answer.
        answers = [(s, extract_final_answer(s)) for s in samples]
        valid = [(s, a) for s, a in answers if a]
        if valid:
            counts = Counter(a for _, a in valid)
            top_answer, top_count = counts.most_common(1)[0]
            agreement = top_count / len(samples)
            matching = [s for s, a in valid if a == top_answer]
            # Among matching samples, prefer the longest (most-detailed) trace.
            chosen = max(matching, key=len)
            return chosen, {
                "strategy": "math-vote",
                "answer": top_answer,
                "agreement": round(agreement, 2),
                "votes": dict(counts),
            }
        # No extractable answer — fall through to longest heuristic.

    if domain == "code":
        # Prefer samples that actually contain a python code block. The verifier
        # will still validate the final pick, but at least we don't choose a
        # purely-prose sample over one with runnable code.
        with_code = [s for s in samples if extract_python(s)]
        if with_code:
            chosen = max(with_code, key=len)
            return chosen, {"strategy": "code-block-present",
                            "agreement": round(len(with_code) / len(samples), 2)}

    # Default: longest answer wins (proxy for "most detailed").
    chosen = max(samples, key=len)
    return chosen, {"strategy": "longest", "agreement": 1.0 / len(samples)}


async def reasoning_step_node(state: GraphState) -> GraphState:
    """One reasoning iteration.

    Iteration 0 — exploration:
      Apply domain-specific system prompt and (for math/code/logic/complex)
      sample N candidates in parallel with the exploration temperature, then
      synthesize a single best response via voting/heuristics.

    Iteration 1+ — refinement:
      Show the model its previous attempt plus the reviewer's critique (and any
      sandbox verification output) at the refinement temperature.
    """
    rid = state["request_id"]
    client = _client()
    thoughts = state["thoughts"]
    iteration = state["iterations"]
    domain = state.get("domain") or "complex"
    cfg = get_domain_config(domain)

    if not thoughts:
        # ----- Exploration ----------------------------------------------------
        msgs = _merge_system_prompt(state["messages"], cfg["system"])

        n_samples = cfg["samples"] if settings.self_consistency_enabled else 1
        n_samples = max(1, min(n_samples, settings.self_consistency_max_samples))
        temp = cfg["exploration_temp"]

        log.info("[%s] REASONING | iteration=1 exploration domain=%s samples=%d temp=%.2f",
                 rid, domain, n_samples, temp)

        async def _one_sample(idx: int) -> str:
            try:
                r = await _api_call(rid, lambda: client.chat.completions.create(
                    model=state["model"],
                    messages=msgs,
                    temperature=temp,
                ))
                # Strip <think>/harmony BEFORE storing — the sample we keep
                # is the user-facing one, otherwise the synthesizer would
                # vote on noise and the next refinement turn would be huge.
                return strip_thinking(r.choices[0].message.content or "").strip()
            except Exception as exc:
                log.warning("[%s] REASONING | sample %d failed: %s", rid, idx, exc)
                return ""

        if n_samples == 1:
            sample = await _one_sample(0)
            samples = [sample] if sample else []
        else:
            results = await asyncio.gather(*[_one_sample(i) for i in range(n_samples)])
            samples = [s for s in results if s]

        if not samples:
            log.error("[%s] REASONING | all %d samples failed", rid, n_samples)
            return {
                **state,
                "thoughts": [""],
                "samples": [],
                "iterations": iteration + 1,
                "final_response": "",
            }

        chosen, info = _synthesize(samples, domain)
        log.info("[%s] REASONING | exploration complete  samples=%d  synth=%s",
                 rid, len(samples), info)
        log.info("[%s] REASONING | chosen preview: %.120s",
                 rid, chosen.replace("\n", " "))

        return {
            **state,
            "thoughts": [chosen],
            "samples": samples,
            "iterations": iteration + 1,
            "final_response": chosen,
        }

    # ----- Refinement ---------------------------------------------------------
    critique = state.get("critique", "The response can be improved.")
    verification = state.get("verification") or {}

    # If the sandbox actually ran and disagreed with the model, surface that as
    # a hard ground-truth signal in the prompt — the reviewer's critique alone
    # is often vague, but stderr/stdout is unambiguous.
    verifier_addon = ""
    if verification and not verification.get("skipped"):
        if not verification.get("ok"):
            verifier_addon = (
                "\n\nA sandbox executed the Python in your previous answer and it FAILED:\n"
                f"  return code: {verification.get('returncode')}\n"
                f"  stderr: {(verification.get('stderr') or '').strip()[:500]}\n"
                f"  stdout: {(verification.get('stdout') or '').strip()[:300]}\n"
                "Diagnose the cause and produce a corrected version."
            )
        else:
            verifier_addon = (
                "\n\nA sandbox executed the Python in your previous answer and it ran cleanly. "
                "Output:\n"
                f"  stdout: {(verification.get('stdout') or '').strip()[:500]}\n"
                "Use this as ground truth when refining."
            )

    log.info("[%s] REASONING | iteration=%d refinement temp=%.2f  critique=%.140s",
             rid, iteration + 1, cfg["refinement_temp"],
             critique.replace("\n", " "))

    msgs = [
        *_merge_system_prompt(state["messages"], cfg["system"]),
        {"role": "assistant", "content": thoughts[-1]},
        {
            "role": "user",
            "content": (
                "A critical reviewer evaluated your response and found these issues:\n\n"
                f"{critique}"
                f"{verifier_addon}\n\n"
                "Provide an improved response that addresses every point above. "
                "Be direct, complete, and self-contained — do not just describe the fix, apply it."
            ),
        },
    ]

    response = await _api_call(rid, lambda: client.chat.completions.create(
        model=state["model"],
        messages=msgs,
        temperature=cfg["refinement_temp"],
    ))

    thought = strip_thinking(response.choices[0].message.content or "").strip()
    log.info("[%s] REASONING | iteration=%d refined  chars=%d  preview: %.100s",
             rid, iteration + 1, len(thought), thought.replace("\n", " "))
    return {
        **state,
        "thoughts": [*thoughts, thought],
        "iterations": iteration + 1,
        "final_response": thought,
        # Clear verification so it gets recomputed on the next pass.
        "verification": {},
    }


async def verify_node(state: GraphState) -> GraphState:
    """Run the latest response through a Python sandbox when the domain asks for it.

    This is the ground-truth signal that the LLM evaluator alone cannot produce.
    A failing exec immediately flips quality to needs_improvement in evaluate_node;
    a passing exec gives the evaluator concrete stdout to compare against.
    """
    rid = state["request_id"]
    domain = state.get("domain") or ""
    cfg = get_domain_config(domain)

    if not settings.sandbox_enabled or not cfg.get("verify"):
        log.info("[%s] VERIFY | skip (enabled=%s domain=%s)",
                 rid, settings.sandbox_enabled, domain)
        return {**state, "verification": {}}

    response = state.get("final_response", "")
    code = extract_python(response)
    if not code:
        log.info("[%s] VERIFY | no python block in response — skip", rid)
        return {**state, "verification": {}}

    log.info("[%s] VERIFY | running sandbox (%d chars, timeout=%.1fs)",
             rid, len(code), settings.sandbox_timeout)

    # subprocess.run is blocking; wrap it so we don't stall the asyncio loop.
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None, lambda: run_python(code, timeout=settings.sandbox_timeout),
    )

    log.info("[%s] VERIFY | ok=%s rc=%s skipped=%s stdout=%d stderr=%d",
             rid, result["ok"], result.get("returncode"),
             result.get("skipped"), len(result.get("stdout", "")),
             len(result.get("stderr", "")))

    return {**state, "verification": result}


async def evaluate_node(state: GraphState) -> GraphState:
    """Evaluate the current response and return a specific, actionable critique.

    If the sandbox ran and FAILED, we short-circuit: no need to ask the LLM
    whether the answer is good — the runtime already said no. The verifier
    output becomes the critique.
    """
    rid = state["request_id"]
    verification = state.get("verification") or {}

    # ---- Sandbox short-circuit ------------------------------------------------
    if verification and not verification.get("skipped") and not verification.get("ok"):
        critique = (
            "The Python in your response failed to execute:\n"
            f"  return code: {verification.get('returncode')}\n"
            f"  stderr: {(verification.get('stderr') or '').strip()[:600]}\n"
            "Fix the bug and produce a corrected, runnable version."
        )
        log.info("[%s] EVALUATE | sandbox failed → NEEDS_WORK (rc=%s)",
                 rid, verification.get("returncode"))
        return {**state, "quality": "needs_improvement", "critique": critique}

    client = _client()
    last_user = _last_user_text(state)

    verifier_note = ""
    if verification and not verification.get("skipped") and verification.get("ok"):
        verifier_note = (
            "\n\nThe Python code in the response was executed successfully. "
            f"stdout (treat as ground truth):\n{(verification.get('stdout') or '').strip()[:600]}"
        )

    prompt = (
        "You are a critical reviewer. Evaluate the response below against the request.\n\n"
        "Reply in this exact format — nothing else:\n"
        "VERDICT: GOOD\n"
        "or\n"
        "VERDICT: NEEDS_WORK\n"
        "CRITIQUE: <bullet list of specific issues and what is missing or wrong>\n\n"
        f"REQUEST:\n{last_user}\n\n"
        f"RESPONSE:\n{state['final_response']}"
        f"{verifier_note}"
    )

    response = await _api_call(rid, lambda: client.chat.completions.create(
        model=state["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ))

    raw = strip_thinking(response.choices[0].message.content or "").strip()
    quality = "good" if "VERDICT: GOOD" in raw.upper() else "needs_improvement"

    log.info("[%s] EVALUATE | verdict=%s", rid, quality.upper())

    critique = ""
    if quality == "needs_improvement":
        # Try single-line CRITIQUE: first
        for line in raw.splitlines():
            if line.upper().startswith("CRITIQUE:"):
                critique = line[len("CRITIQUE:"):].strip()
                break
        # If not found / empty, treat CRITIQUE: as a section header.
        if not critique:
            lines = raw.splitlines()
            in_critique = False
            parts = []
            for line in lines:
                if line.upper().startswith("CRITIQUE:"):
                    in_critique = True
                    parts.append(line[len("CRITIQUE:"):].strip())
                elif in_critique:
                    parts.append(line)
            critique = "\n".join(parts).strip()

        log.info("[%s] EVALUATE | critique: %s", rid, critique[:400])

    return {**state, "quality": quality, "critique": critique}


async def stream_final_node(state: GraphState) -> GraphState:
    """Deliver the reasoned response to the user.

    No extra LLM call — the reasoning already produced a good answer (EVALUATE=GOOD).
    We strip internal reasoning tags and stream the result directly, avoiding
    context-size issues and the latency of a second generation.
    """
    response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    clean = strip_thinking(state["final_response"])

    log.info("[%s] STREAM_FINAL | after %d iteration(s)  stream=%s  chars=%d",
             state["request_id"], state["iterations"], state["do_stream"], len(clean))

    if state["do_stream"]:
        # Send the complete response as a single SSE chunk — the user already waited
        # for the reasoning loop; there is no benefit in simulating token-by-token
        # streaming here, and chunked loops stall the asyncio event loop.
        await _push(state["request_id"], _sse_chunk(clean, state["model"], response_id))
        await _push(state["request_id"], _sse_chunk("", state["model"], response_id, finish=True))
        await _close(state["request_id"])
        log.info("[%s] STREAM_FINAL | pushed to queue — done", state["request_id"])

    return {**state, "final_response": clean}
