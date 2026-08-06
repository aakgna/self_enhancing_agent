"""Check-in agent: same-day patient check-in against a local sample EHR.

Follows the standard agent contract (see general_qa/agent.py for the full
statement of the contract and the write rule). This is the first genuinely
multi-turn agent in this codebase: all flow state lives in
agent_scratchpad["checkin"] (the only place this agent may write). While a
flow is incomplete it sets awaiting_reply=True, which is what makes
route_node keep the conversation with this agent turn after turn instead of
reclassifying a short reply like a spoken DOB from scratch (see
orchestrator/supervisor.py::route_node's stickiness step).

Ported from FurHat/app/agent/specialists/checkin.py, but conversational-only:
no camera/ID-card OCR, no insurance-card scan, no tablet dashboard — identity
is confirmed by spoken/typed name + date of birth against a local sample EHR
(tools/shared/ehr_csv). Scheduling a new appointment and insurance coverage
detail lookups are explicitly out of scope for this agent (and this domain).

Deterministic work — step transitions, EHR lookups, retry limits, and yes/no
parsing (FurHat's own _classify_yes_no keyword sets, ported verbatim) — lives
in plain, directly-testable functions. The llm is used only to phrase each
step's reply from facts already decided in code, the same "deterministic
logic decides, llm renders" split gmail_recent_reader uses: it is given the
exact facts to communicate and told not to add or omit any of them, so it
can't invent details in a flow that touches consent and identity.
"""

from pydantic import BaseModel

from langchain_core.messages import HumanMessage, SystemMessage

from tools.shared.ehr_csv import store

AGENT_ID = "checkin"
AGENT_VERSION = 1

DEFAULT_PROMPT = (
    "You are the front-desk check-in specialist at a clinic. You will be told "
    "exactly what to say at each step, including any patient details already "
    "confirmed. Rephrase it warmly and concisely in 1-2 sentences — do not add "
    "or omit any fact you were given, and never invent a name, time, or detail "
    "that wasn't provided to you."
)

# Ported verbatim from FurHat/app/agent/orchestrator.py — a deterministic
# yes/no reply classifier, used here for consent and forms confirmations so
# a compliance-sensitive answer is never left to model interpretation.
_AFFIRMATIVE_WORDS = {
    "yes", "yeah", "yep", "yup", "sure", "ok", "okay", "please",
    "sounds good", "let's do it", "definitely", "absolutely",
}
_NEGATIVE_WORDS = {"no", "nope", "nah", "not now", "no thanks", "no thank you", "later"}

_MAX_IDENTITY_ATTEMPTS = 2


def _classify_yes_no(text: str) -> str | None:
    """Return 'yes', 'no', or None if the reply isn't clearly affirmative/negative."""
    normalized = text.lower().strip().rstrip(".!?")
    if not normalized:
        return None
    if normalized in _AFFIRMATIVE_WORDS or any(normalized.startswith(w + " ") for w in _AFFIRMATIVE_WORDS):
        return "yes"
    if normalized in _NEGATIVE_WORDS or any(normalized.startswith(w + " ") for w in _NEGATIVE_WORDS):
        return "no"
    return None


class IdentityFields(BaseModel):
    first_name: str
    last_name: str
    dob: str  # YYYY-MM-DD


def _latest_human_text(state: dict) -> str:
    for message in reversed(state["messages"]):
        if getattr(message, "type", None) == "human":
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def _render(llm, system_prompt: str, instruction: str):
    return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=instruction)])


def _step(reply, scratch: dict) -> dict:
    return {"messages": [reply], "agent_scratchpad": {AGENT_ID: scratch}}


def run(state: dict, system_prompt: str, llm) -> dict:
    text = _latest_human_text(state)
    own = state["agent_scratchpad"].get(AGENT_ID, {})
    step = own.get("step", "start")

    if step == "awaiting_identity":
        extractor = llm.with_structured_output(IdentityFields, method="json_schema")
        fields = extractor.invoke(
            "Extract the patient's first name, last name, and date of birth "
            f'(YYYY-MM-DD) from this reply to "what is your full name and date '
            f'of birth": {text}'
        )
        patient = store.find_patient(fields.first_name, fields.last_name, fields.dob)
        attempts = own.get("attempts", 0) + 1

        if patient is None:
            if attempts >= _MAX_IDENTITY_ATTEMPTS:
                reply = _render(
                    llm, system_prompt,
                    "Tell the patient you still can't find a record matching "
                    "what they gave you, and ask them to check in with a staff "
                    "member at the desk instead.",
                )
                return _step(reply, {"step": "escalated", "awaiting_reply": False})
            reply = _render(
                llm, system_prompt,
                "Tell the patient you couldn't find a record with that name "
                "and date of birth, and ask them to double check the spelling "
                "and try again.",
            )
            return _step(reply, {"step": "awaiting_identity", "awaiting_reply": True, "attempts": attempts})

        appointment = store.find_appointment_today(patient["patient_id"])
        if appointment is None:
            reply = _render(
                llm, system_prompt,
                f"Tell {patient['first_name']} you don't see an appointment "
                "scheduled for today, and suggest they check with a staff "
                "member if they believe that's a mistake.",
            )
            return _step(
                reply,
                {"step": "no_appointment", "awaiting_reply": False, "patient_id": patient["patient_id"]},
            )

        reply = _render(
            llm, system_prompt,
            f"Tell {patient['first_name']} you see them scheduled today at "
            f"{appointment['time']} for {appointment['reason']}, and ask them "
            "to confirm that's right.",
        )
        return _step(
            reply,
            {
                "step": "awaiting_appointment_ack",
                "awaiting_reply": True,
                "patient_id": patient["patient_id"],
                "appointment_id": appointment["appointment_id"],
            },
        )

    if step == "awaiting_appointment_ack":
        if _classify_yes_no(text) != "yes":
            reply = _render(
                llm, system_prompt,
                "Tell the patient no problem, and ask them to check with a "
                "staff member at the desk so the right appointment can be "
                "sorted out.",
            )
            return _step(reply, {**own, "step": "escalated", "awaiting_reply": False})
        reply = _render(llm, system_prompt, "Ask the patient if they consent to receive treatment today.")
        return _step(reply, {**own, "step": "awaiting_consent", "awaiting_reply": True})

    if step == "awaiting_consent":
        if _classify_yes_no(text) != "yes":
            reply = _render(
                llm, system_prompt,
                "Tell the patient that's okay, and let them know a staff "
                "member will come speak with them about it before continuing.",
            )
            return _step(reply, {**own, "step": "escalated", "awaiting_reply": False})
        reply = _render(llm, system_prompt, "Ask the patient if they've completed their intake forms.")
        return _step(reply, {**own, "step": "awaiting_forms", "awaiting_reply": True})

    if step == "awaiting_forms":
        forms_done = _classify_yes_no(text) == "yes"
        instruction = (
            "Tell the patient they're all checked in and ask them to have a "
            "seat, they'll be called shortly."
            if forms_done
            else
            "Tell the patient they're checked in, and that a staff member will "
            "bring them the remaining intake forms — ask them to have a seat."
        )
        reply = _render(llm, system_prompt, instruction)
        return _step(reply, {**own, "step": "complete", "awaiting_reply": False, "forms": "completed" if forms_done else "pending"})

    # step is "start" (fresh session), or a terminal state from a prior
    # check-in ("complete" / "escalated" / "no_appointment") being started
    # over for a new request: begin a fresh check-in.
    reply = _render(llm, system_prompt, "Ask the patient for their full name and date of birth to check in.")
    return _step(reply, {"step": "awaiting_identity", "awaiting_reply": True, "attempts": 0})
