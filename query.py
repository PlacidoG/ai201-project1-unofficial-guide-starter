"""Grounded generation (Milestone 5).

End-to-end RAG question answering: retrieve the most relevant professor-review
chunks from ChromaDB (embed_index.retrieve), build a grounded prompt, and ask
Llama 3.3 70B via Groq to answer using ONLY those reviews.

For in-depth questions (quality, grading, difficulty, "best/worst", comparisons)
the answer is backed by one supporting example review per professor named in the
answer. For simple factual lookups (who teaches a course, how many professors)
no example reviews are shown.

    from query import ask
    result = ask("Which professor is most lenient with late work?")
    print(result["answer"])
    for ex in result["examples"]:
        print(ex["professor"], ex["text"])
    print(result["sources"])

Requires a GROQ_API_KEY in .env (copy .env.example to .env). Groq's free tier
needs no credit card: https://console.groq.com
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from groq import Groq, APIError, AuthenticationError

from embed_index import retrieve, TOP_K

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"   # Llama 3.3 70B via Groq, per diagram.JPG

# We retrieve top_k (=40, per planning.md) for recall, but only send the strongest
# LLM_CONTEXT_K chunks to the model. On this small corpus top-40 is nearly the whole
# dataset (~5k tokens/query); trimming the LLM context keeps us well under Groq's
# free-tier token limits without changing the retrieval spec.
LLM_CONTEXT_K = 12

SYSTEM_PROMPT = (
    "You are The Unofficial Guide, an assistant that answers questions about Computer "
    "Science professors at the University of Houston-Downtown using ONLY the student "
    "reviews provided as context. Each review in the context is prefixed with a [N] "
    "citation number.\n"
    "\n"
    "Answer rules:\n"
    "- Base every claim strictly on the provided reviews; no outside knowledge. If the "
    "reviews lack the information, say so plainly.\n"
    "- Name the relevant professors and attribute every claim to them by name.\n"
    "- SYNTHESIZE. Never restate reviews one by one and NEVER answer in a single prose "
    "paragraph. Group reviews that touch the same point into one THEME and state the "
    "consensus or the disagreement.\n"
    "- Format EVERY in-depth answer as labeled theme lines with a blank line between "
    "them, following this shape exactly:\n"
    "    Top picks: Pakhrin, Yilmaz, Zhang (all rated 5/5).\n"
    "    Teaching clarity: Pakhrin and Kamto explain concepts clearly; Yilmaz is "
    "conceptual but easy to follow.\n"
    "    Grading fairness: Yilmaz and Pakhrin grade fairly with clear exam expectations; "
    "opinions are split on Harris.\n"
    "    Workload/difficulty: mostly low-to-moderate (2-3/5) for the top picks.\n"
    "  (The leading 'Top picks' line is only for ranking / 'top N' questions.)\n"
    "- Choose theme labels that directly answer the asked question (e.g. Grading "
    "fairness, Teaching clarity, Workload/difficulty, Exam alignment, "
    "Lateness/flexibility) and fold in the concrete fields that support them (ratings "
    "out of 5, difficulty out of 5, course codes).\n"
    "- For simple factual questions (who teaches a course, counts, plain lists), skip "
    "the themes and give a direct one- or two-line answer.\n"
    "- Keep every line short and specific — phrases, not long sentences.\n"
    "\n"
    "Respond with a SINGLE JSON object of this exact shape (no prose outside it):\n"
    '{"answer": "<your concise grounded answer>", '
    '"needs_examples": <true or false>, '
    '"examples": [{"professor": "<exact professor name>", "review": <citation N>}]}\n'
    "\n"
    "Set needs_examples to FALSE for simple factual lookups — e.g. which professor "
    "teaches a course, how many professors there are, or just listing names — and "
    "return an empty examples list.\n"
    "Set needs_examples to TRUE for in-depth questions about quality, grading, "
    "difficulty, workload, comparisons, or best/worst judgments.\n"
    "When needs_examples is TRUE: for EACH professor you name in the answer, include "
    "exactly ONE entry in examples — the single citation number whose review best "
    "supports what you said about that professor and best matches the question. If the "
    "question targets a specific course, pick a review for that course. Never include "
    "more than one review per professor, never repeat a review, and never include a "
    "professor you did not name in the answer. Each 'review' value must be one of the "
    "[N] citation numbers shown in the context."
)

_client: Groq | None = None


def get_client() -> Groq:
    """Build (and cache) the Groq client from the GROQ_API_KEY in the environment."""
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example to .env and add your free "
                "Groq key from https://console.groq.com"
            )
        _client = Groq(api_key=api_key)
    return _client


def format_context(hits: list[dict]) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        m = hit["metadata"]
        blocks.append(
            f"[{i}] {m.get('professor_name')} - {m.get('course')} "
            f"(rating {m.get('rating_overall')}/5, difficulty {m.get('difficulty')}/5, "
            f"{m.get('date')}, via {m.get('source')}):\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def format_sources(hits: list[dict]) -> list[str]:
    """Distinct, human-readable provenance lines for the retrieved chunks.

    Deduplicated by (professor, course, date) while preserving retrieval order.
    """
    seen: set[tuple] = set()
    sources: list[str] = []
    for hit in hits:
        m = hit["metadata"]
        key = (m.get("professor_name"), m.get("course"), m.get("date"))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            f"{m.get('professor_name')} - {m.get('course')} "
            f"(rating {m.get('rating_overall')}/5, difficulty {m.get('difficulty')}/5, "
            f"{m.get('date')}) [{m.get('source')}]"
        )
    return sources


def _citation_number(value) -> int | None:
    """Coerce a model-supplied citation to an int (handles 8, '8', '[8]')."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return int(match.group())
    return None


def _professor_in_answer(professor: str | None, answer: str) -> bool:
    """True if a named professor is actually mentioned in the answer text."""
    if not professor or professor.lower() == "unknown":
        return False
    answer_l = answer.lower()
    if professor.lower() in answer_l:
        return True
    last_name = professor.split()[-1].lower()
    return re.search(rf"\b{re.escape(last_name)}\b", answer_l) is not None


def _answer_position(professor: str, answer: str) -> int:
    """Index of a professor's first mention in the answer (for ordering)."""
    answer_l = answer.lower()
    pos = answer_l.find(professor.lower())
    if pos == -1:
        pos = answer_l.find(professor.split()[-1].lower())
    return pos if pos != -1 else len(answer)


def select_examples(raw_examples: list[dict], hits: list[dict], answer: str) -> list[dict]:
    """Map the model's citation picks to real reviews, deduped one-per-professor.

    Keeps only named professors that actually appear in the answer, at most one
    review each, ordered to match the order they appear in the answer.
    """
    selected: list[dict] = []
    seen_professors: set[str] = set()
    for item in raw_examples or []:
        num = _citation_number(item.get("review"))
        if num is None or not (1 <= num <= len(hits)):
            continue
        meta = hits[num - 1]["metadata"]
        professor = meta.get("professor_name")
        if professor in seen_professors:                 # one review per professor
            continue
        if not _professor_in_answer(professor, answer):  # only professors in the answer
            continue
        seen_professors.add(professor)
        selected.append({
            "professor": professor,
            "course": meta.get("course"),
            "rating_overall": meta.get("rating_overall"),
            "difficulty": meta.get("difficulty"),
            "date": meta.get("date"),
            "text": hits[num - 1]["text"],
        })
    selected.sort(key=lambda e: _answer_position(e["professor"], answer))
    return selected


def tidy_answer(text: str) -> str:
    """Clean up the model's themed answer: flush-left lines, single blank-line gaps."""
    out: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            if out and out[-1] == "":   # collapse runs of blank lines
                continue
            out.append("")
        else:
            out.append(stripped)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def format_examples(examples: list[dict]) -> str:
    """Render supporting example reviews for display, organized per professor."""
    blocks = []
    for ex in examples:
        header = (
            f"{ex['professor']} - {ex['course']} "
            f"(rating {ex['rating_overall']}/5, difficulty {ex['difficulty']}/5, "
            f"{ex['date']})"
        )
        blocks.append(f"{header}\n\"{ex['text']}\"")
    return "\n\n".join(blocks)


def ask(question: str, top_k: int = TOP_K, llm_context_k: int = LLM_CONTEXT_K) -> dict:
    """Answer `question` from the retrieved reviews.

    Returns {"answer": str, "examples": list[dict], "sources": list[str]}.
    Retrieves `top_k` chunks but only sends the strongest `llm_context_k` of them
    to the LLM; `sources` and `examples` reflect those chunks. `examples` is empty
    for simple factual questions and otherwise holds one supporting review per
    professor named in the answer.
    """
    if not question or not question.strip():
        return {"answer": "Please enter a question.", "examples": [], "sources": []}

    hits = retrieve(question, top_k=top_k)
    if not hits:
        return {
            "answer": "No reviews are indexed yet. Run `python embed_index.py` first.",
            "examples": [],
            "sources": [],
        }

    # Only the strongest chunks go to the model; [N] citations index into this subset.
    context_hits = hits[:llm_context_k]
    user_message = (
        f"Context — student reviews:\n\n{format_context(context_hits)}\n\n"
        f"Question: {question}"
    )
    sources = format_sources(context_hits)
    try:
        response = get_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except AuthenticationError:
        return {
            "answer": "Groq rejected the API key (401). Put a valid GROQ_API_KEY in "
                      ".env — get a free one at https://console.groq.com",
            "examples": [],
            "sources": sources,
        }
    except APIError as e:
        return {"answer": f"Groq API error: {e}", "examples": [], "sources": sources}

    content = response.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        answer = tidy_answer(data.get("answer") or "")
        needs_examples = bool(data.get("needs_examples"))
        raw_examples = data.get("examples") or []
    except (json.JSONDecodeError, AttributeError):
        # Model didn't return valid JSON — fall back to showing the raw text.
        return {"answer": tidy_answer(content), "examples": [], "sources": sources}

    examples = select_examples(raw_examples, context_hits, answer) if needs_examples else []
    return {"answer": answer, "examples": examples, "sources": sources}


if __name__ == "__main__":
    for demo_q in (
        "What professors are the best for taking CS core classes?",   # in-depth
        "What professors teach CS1411?",                              # simple lookup
    ):
        out = ask(demo_q)
        print("=" * 80)
        print("Q:", demo_q, "\n")
        print(out["answer"], "\n")
        if out["examples"]:
            print("Supporting reviews:")
            print(format_examples(out["examples"]), "\n")
        print("Retrieved from:", len(out["sources"]), "sources")
