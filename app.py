import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
from markupsafe import Markup, escape

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-only-change-in-production")

# Unversioned aliases (e.g. gemini-1.5-flash) are often retired → 404. We pick a working id from the API unless GEMINI_MODEL is set.
_FALLBACK_MODEL = "gemini-2.5-flash"
_RESOLVED_MODEL: str | None = None
_MAX_MESSAGES = 24

_SYSTEM_INSTRUCTION = """You are a careful medical information assistant powered by a large language model.

Scope and tone:
- Give clear, evidence-aligned general medical information in plain language.
- Prefer structured answers with short sections when helpful: Overview, Possible causes, What to watch for, Self-care (if appropriate), When to seek urgent care, Follow-up questions for a clinician.
- Never claim to have examined the user or seen their records.

Safety:
- If the user describes possible emergency symptoms (e.g. chest pain, stroke signs, severe bleeding, trouble breathing, loss of consciousness, thoughts of self-harm), tell them to call local emergency services immediately and not rely on this chat.
- Do not prescribe specific drug doses or start/stop medications; encourage a licensed clinician for prescribing decisions.
- State uncertainty when guidelines conflict or evidence is limited.

You are not a substitute for a licensed healthcare professional."""


def _configure_genai() -> None:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return
    genai.configure(api_key=key)


def _list_generate_content_model_ids() -> list[str]:
    """Return short model ids (e.g. gemini-2.5-flash) that support generateContent for this API key."""
    ids: list[str] = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" not in methods:
                continue
            raw = getattr(m, "name", "") or ""
            short = raw.split("/", 1)[-1] if "/" in raw else raw
            if short:
                ids.append(short)
    except Exception:
        pass
    return ids


def _pick_model_id() -> str:
    """Use GEMINI_MODEL if set; otherwise the best match from ListModels (avoids 404 on retired names)."""
    global _RESOLVED_MODEL
    explicit = (os.getenv("GEMINI_MODEL") or "").strip()
    if explicit:
        return explicit
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    candidates = _list_generate_content_model_ids()
    # Prefer current Flash workhorses; skip live/TTS/embedding.
    def usable(name: str) -> bool:
        low = name.lower()
        if "embedding" in low or "tts" in low:
            return False
        if "live" in low and "flash" in low:
            return False
        return True

    candidates = [c for c in candidates if usable(c)]
    prefer_in_order = (
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest",
    )
    chosen = ""
    for needle in prefer_in_order:
        for c in candidates:
            if needle in c.lower():
                chosen = c
                break
        if chosen:
            break
    if not chosen and candidates:
        chosen = next((c for c in candidates if "flash" in c.lower()), candidates[0])
    if not chosen:
        chosen = _FALLBACK_MODEL
    _RESOLVED_MODEL = chosen
    return chosen


def effective_model_name() -> str:
    """Label for UI: explicit env, or auto-resolved id, or fallback string before first API call."""
    explicit = (os.getenv("GEMINI_MODEL") or "").strip()
    if explicit:
        return explicit
    if _RESOLVED_MODEL:
        return _RESOLVED_MODEL
    if not os.getenv("GOOGLE_API_KEY"):
        return _FALLBACK_MODEL
    return _pick_model_id()


def _get_model():
    _configure_genai()
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    return genai.GenerativeModel(
        model_name=_pick_model_id(),
        system_instruction=_SYSTEM_INSTRUCTION,
    )


def _part_to_text(part: Any) -> str:
    if part is None:
        return ""
    t = getattr(part, "text", None)
    if t:
        return str(t)
    return str(part)


def _response_text(response: Any) -> str:
    try:
        return (response.text or "").strip()
    except ValueError:
        pass
    chunks: list[str] = []
    for cand in getattr(response, "candidates", []) or []:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for p in parts:
            chunks.append(_part_to_text(p))
    out = "\n".join(s for s in chunks if s).strip()
    return out or "The model did not return readable text. Try a shorter question or try again."


def _trim_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(messages) <= _MAX_MESSAGES:
        return messages
    return messages[-_MAX_MESSAGES:]


def _to_gemini_history(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        out.append({"role": role, "parts": [m["content"]]})
    return out


def format_plain_response(text: str) -> Markup:
    safe = escape(text or "")
    return Markup(safe.replace("\n", "<br>\n"))


app.jinja_env.filters["nl2br"] = format_plain_response


@app.route("/", methods=["GET"])
def index():
    messages = session.get("messages") or []
    return render_template(
        "index.html",
        messages=messages,
        error=session.pop("error", None),
        model_name=effective_model_name(),
        has_api_key=bool(os.getenv("GOOGLE_API_KEY")),
    )


@app.route("/chat", methods=["POST"])
def chat():
    user_text = (request.form.get("message") or "").strip()
    if not user_text:
        session["error"] = "Please enter a message."
        return redirect(url_for("index"))

    if not os.getenv("GOOGLE_API_KEY"):
        session["error"] = "Missing GOOGLE_API_KEY. Add it to your environment or a .env file."
        return redirect(url_for("index"))

    model = _get_model()
    if model is None:
        session["error"] = "Could not initialize the model. Check GOOGLE_API_KEY."
        return redirect(url_for("index"))

    messages = _trim_messages(list(session.get("messages") or []))
    messages.append({"role": "user", "content": user_text})

    prior = messages[:-1]
    try:
        gemini_history = _to_gemini_history(prior)
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_text)
        reply = _response_text(response)
    except Exception as exc:  # noqa: BLE001 — surface API errors to the user
        session["error"] = f"The assistant could not complete a reply: {exc}"
        return redirect(url_for("index"))

    messages.append({"role": "assistant", "content": reply})
    session["messages"] = _trim_messages(messages)
    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear():
    session.pop("messages", None)
    session.pop("error", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
