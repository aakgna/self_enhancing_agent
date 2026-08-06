"""Domain scope gate: "is this request even something the system should try to
handle" — checked before registry search or generation is ever attempted.

Structured the same way as classifier.py: a small, fast model call over a
structured pydantic verdict. Unlike the classifier (which decides *which*
agent), this decides *whether the system should act at all*. Declining here
is final for the turn — no registry search, no capability-gap logging, no
agent generation.

The domain definition itself (core function statement + concrete in-scope and
out-of-scope examples) is not hardcoded here — it's versioned, editable text
stored via db.prompt_store under the reserved id "_scope" (see chat.py's
seeding call and the :scope REPL command), so the boundary can be changed live
without a redeploy, same as any agent's prompt.
"""

from pydantic import BaseModel, Field

from orchestrator.llm import fast_llm

DEFAULT_SCOPE_PROMPT = """\
CORE FUNCTION: This system operates as a clinic/hospital front-desk assistant. \
Its legitimate purposes are:
  1. Same-day patient check-in (confirming identity, today's appointment, \
consent, and intake forms).
  2. Scheduling: booking, rescheduling, or canceling a patient appointment.
  3. Answering general facility, treatment, billing, and administrative \
questions a patient might ask at a front desk (hours, location, parking, \
insurance accepted, what to bring, policies, etc.).

Not every legitimate purpose has a pre-built agent yet — that's expected and \
fine: an in-scope request with no existing agent should be treated as a gap \
to fill (build a new agent), not declined. Only decline requests that fall \
outside these three purposes entirely.

IN-SCOPE examples:
  - "I'm here for my 2pm appointment, my name is..."
  - "What are your office hours?"
  - "Do you accept my insurance?"
  - "Where do I park?"
  - "I need to check in."
  - "What should I bring to my appointment?"
  - "Can you schedule a follow-up appointment for me next month?"
  - "What's the next available appointment with Dr. Foster?"
  - "I need to reschedule my Tuesday appointment."

OUT-OF-SCOPE examples (decline these, even though the system could technically \
build an agent for them):
  - "Check my email for anything urgent." (no email capability in this domain)
  - "What's 3 times the derivative of x^2?" (no general math/calculus capability)
  - "Convert this text to pig latin." (no text-transformation capability)
  - "Translate this into French."
  - "What's my bank balance?" / any banking or financial request.
  - "Can you check if my insurance covers an MRI?" (coverage *details* are out \
of scope; confirming insurance is on file during check-in is in scope)
  - Any general trivia, coding, writing, or entertainment request unrelated to \
a front-desk visit.
"""


class ScopeVerdict(BaseModel):
    in_scope: bool = Field(
        description="True if the request is squarely front-desk check-in, "
        "appointment scheduling, or facility/treatment/billing FAQ — even if "
        "no existing agent covers it yet, that's a gap to fill, not a reason "
        "to decline. False only for requests outside all three purposes "
        "entirely, even if the system could technically build an agent for it."
    )
    reason: str = Field(
        description="One short, polite sentence. If in_scope, a brief note "
        "(not shown to the user). If out of scope, this IS the reply spoken/"
        "printed back to the user declining the request."
    )


_PROMPT = """\
You are a scope gate for a domain-constrained assistant. Decide whether the \
user's request falls inside the assistant's declared domain, below. Be strict: \
the assistant must refuse anything outside its domain even if it could \
technically attempt it.

{scope_config}

User request: {request}

If out of scope, `reason` must be a short, polite, in-character decline (e.g. \
"I'm only able to help with check-in and clinic questions here — I can't help \
with that.") — it will be shown to the user verbatim."""


def check(request: str, scope_config: str, llm=None) -> ScopeVerdict:
    chooser = (llm or fast_llm()).with_structured_output(ScopeVerdict)
    return chooser.invoke(_PROMPT.format(scope_config=scope_config, request=request))
