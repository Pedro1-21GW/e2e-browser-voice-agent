---
name: feynman-explain
description: Explain a system, finding, or investigation result using the Feynman technique — plain language, concrete analogies, no unexplained jargon. Use when the user says "explain like Feynman", "Feynman method", "explain simply", "ELI5", or asks to understand *why* something works/doesn't work rather than just *what* the finding is.
---

# Feynman-style explanation

The goal is not to summarize findings — it's to make the reader actually understand
the mechanism, the way Richard Feynman explained physics: strip jargon, use a concrete
analogy, and expose the underlying "why" so a non-specialist could rebuild the intuition
themselves.

## Method

1. **State the plain-English core idea first**, before any details. One or two sentences,
   no acronyms, as if explaining to a smart person outside the field.
2. **Use one concrete, physical analogy** that maps closely to the real mechanism (moving
   truck for memory budgets, recipe/kitchen for setup pipelines, plumbing for data flow,
   etc.). Pick the analogy that makes the *actual numbers or constraints* click, not just
   a decorative metaphor.
3. **Do the real math/facts inline, in the analogy's terms**, so the reader can verify the
   conclusion themselves rather than trusting an assertion. E.g. don't just say "not enough
   VRAM" — show the truck-size vs furniture-size numbers so the gap is obvious.
4. **Name the surprising part explicitly.** If reality differs from what the docs/README/
   plan claim (nothing cloned yet, wrong precision installed, a step silently skipped),
   call that out as the headline, not a footnote — that's usually the actual answer to
   what the user is asking.
5. **End with a one-line bottom-line verdict** in plain language: does it work, can it run,
   is it up to date — whatever the user's actual question was — stated directly, not hedged.

## What to avoid

- Don't restate documentation or config verbatim — translate it.
- Don't pile on multiple analogies; one well-chosen analogy beats three.
- Don't bury the surprising/blocking finding under a wall of confirmed-OK details.
- Don't skip the arithmetic — a Feynman explanation earns trust by showing the numbers,
  not by asserting a conclusion.

## When investigating before explaining

If the question requires checking real state (repo contents, installed versions, hardware,
config files) rather than just re-explaining known context, do that investigation first
(read files, run version/hardware checks) — the explanation is only as good as the facts
underneath it. Prefer verified facts ("torch reports cuda available: False") over inferred
ones ("probably CPU-only").
