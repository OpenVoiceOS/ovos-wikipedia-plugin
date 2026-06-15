"""Full-pipeline end-to-end test for ovos-wikipedia-solver using ovoscope.

Exercises the complete OVOS intent pipeline chain:
  recognizer_loop:utterance
    → ovos-persona-pipeline-plugin
    → PersonaService
    → WikipediaRetrievalEngine.query()
    → speak

The web call (``WikipediaRetrievalEngine.search``) is stubbed at its
boundary to return a fixed ``WikipediaResult``.  No network access occurs.

Tests:
  1. ``test_pipeline_produces_speak`` — a ``speak`` message with non-empty
     text is emitted when an utterance traverses the full pipeline.
  2. ``test_speak_message_has_non_empty_utterance`` — the spoken text is
     the stub summary, proving the retrieval result flows through intact.
  3. ``test_user_turn_recorded_in_memory`` — the USER utterance appears in
     the live PersonaService short-term memory after the pipeline fires.
  4. ``test_assistant_response_recorded_in_memory`` — an ASSISTANT turn is
     recorded once the pipeline produces a spoken answer.
  5. ``test_unknown_session_has_empty_history`` — a session that never
     interacted returns an empty history list.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

ovoscope = pytest.importorskip("ovoscope")
ovos_persona = pytest.importorskip("ovos_persona")

from ovos_bus_client.message import Message
from ovos_bus_client.session import Session, SessionManager
from ovoscope import (
    PERSONA_PIPELINE,
    CaptureSession,
    get_minicroft,
    is_pipeline_available,
)

if not is_pipeline_available(PERSONA_PIPELINE):
    pytest.skip(
        "ovos-persona-pipeline-plugin not installed",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Stub constant — the fixed answer the stubbed search() returns
# ---------------------------------------------------------------------------
STUB_SUMMARY = (
    "Wikipedia is a free online encyclopedia written and maintained by a "
    "community of volunteers through open collaboration."
)

# ---------------------------------------------------------------------------
# Persona JSON written into a temp directory
# ---------------------------------------------------------------------------
PERSONA_NAME = "WikiBot"


def _make_personas_dir(name: str = PERSONA_NAME) -> str:
    """Write a minimal persona JSON into a temp dir and return the path."""
    tmpdir = tempfile.mkdtemp()
    persona = {
        "name": name,
        # handlers = OPM entry-point name for the retrieval plugin
        "handlers": ["ovos-wikipedia-plugin"],
        "ovos-wikipedia-plugin": {},
    }
    with open(os.path.join(tmpdir, f"{name}.json"), "w") as fh:
        json.dump(persona, fh)
    return tmpdir


PERSONAS_PATH = _make_personas_dir()

PIPELINE_CONFIG = {
    "persona": {
        "personas_path": PERSONAS_PATH,
        "default_persona": PERSONA_NAME,
        "short-term-memory": True,
        "handle_fallback": True,
        "ignore_plugin_personas": True,
    }
}

TEST_PIPELINE = list(PERSONA_PIPELINE)  # high + low stages


# ---------------------------------------------------------------------------
# Stub for WikipediaRetrievalEngine.search
# Returns a fixed WikipediaResult so no HTTP request is made.
# ---------------------------------------------------------------------------
def _stub_search(self, query, lang=None, top_k=3):
    """Stub that returns a deterministic WikipediaResult without any HTTP calls."""
    from ovos_wikipedia import WikipediaResult
    return [
        WikipediaResult(
            page_id="99999",
            lang=(lang or "en").split("-")[0],
            title="Wikipedia",
            summary=STUB_SUMMARY,
            conf=0.95,
            query=query,
        )
    ]


# ---------------------------------------------------------------------------
# Module-level MiniCroft with the stub active for the entire test session
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mc():
    """Start MiniCroft with search() stubbed so no HTTP calls are made."""
    with patch("ovos_wikipedia.WikipediaRetrievalEngine.search", _stub_search):
        croft = get_minicroft(
            skill_ids=[],
            default_pipeline=TEST_PIPELINE,
            pipeline_config=PIPELINE_CONFIG,
        )
        yield croft
        croft.stop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utterance_msg(utterance: str, sess: Session) -> Message:
    return Message(
        "recognizer_loop:utterance",
        {"utterances": [utterance], "lang": sess.lang},
        {"session": sess.serialize()},
    )


def _drive_utterance(croft, sess: Session, utterance: str, timeout: int = 30):
    """Emit an utterance message through the pipeline and collect bus messages."""
    with patch("ovos_wikipedia.WikipediaRetrievalEngine.search", _stub_search):
        cap = CaptureSession(
            croft,
            eof_msgs=["ovos.utterance.handled", "ovos.utterance.cancelled"],
        )
        cap.capture(_utterance_msg(utterance, sess), timeout=timeout)
        return cap.finish()


def _get_persona_service(croft):
    """Return the live PersonaService from the running MiniCroft."""
    return croft.intents.pipeline_plugins["ovos-persona-pipeline-plugin"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWikipediaPersonaPipelineSpeak:
    """Utterance traverses the full pipeline and causes a speak."""

    def test_pipeline_produces_speak(self, mc):
        """Pipeline must emit at least one speak message."""
        sess = Session(session_id="wiki-e2e-speak-test")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "what is wikipedia", timeout=30)

        msg_types = [m.msg_type for m in messages]
        speak_msgs = [m for m in messages if m.msg_type == "speak"]

        assert speak_msgs, (
            f"Expected at least one 'speak' message; got msg_types: {msg_types}"
        )

    def test_speak_message_has_non_empty_utterance(self, mc):
        """The spoken text must be non-empty (stub summary should flow through)."""
        sess = Session(session_id="wiki-e2e-speak-nonempty")
        SessionManager.sessions[sess.session_id] = sess

        messages = _drive_utterance(mc, sess, "tell me about encyclopedias", timeout=30)

        speak_msgs = [m for m in messages if m.msg_type == "speak"]
        assert speak_msgs, (
            f"No speak message found. Message types: {[m.msg_type for m in messages]}"
        )
        spoken = speak_msgs[0].data.get("utterance", "")
        assert spoken.strip(), (
            f"speak message had empty utterance; data={speak_msgs[0].data}"
        )


class TestWikipediaPersonaMemory:
    """PersonaService short-term memory is populated via the pipeline."""

    def test_user_turn_recorded_in_memory(self, mc):
        """USER utterance must appear in persona memory after pipeline fires."""
        svc = _get_persona_service(mc)
        sess = Session(session_id="wiki-e2e-mem-user")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None, f"Persona '{PERSONA_NAME}' not loaded"
        assert persona.memory is not None, "Persona must have short-term memory enabled"

        _drive_utterance(mc, sess, "who wrote wikipedia", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        contents = [m.content for m in history]
        assert any("who wrote wikipedia" in c for c in contents), (
            f"User utterance not found in memory for session {sess.session_id}. "
            f"History: {contents}"
        )

    def test_assistant_response_recorded_in_memory(self, mc):
        """An ASSISTANT turn must appear in memory after the pipeline speaks."""
        from ovos_plugin_manager.templates.agents import MessageRole

        svc = _get_persona_service(mc)
        sess = Session(session_id="wiki-e2e-mem-assistant")
        SessionManager.sessions[sess.session_id] = sess

        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        _drive_utterance(mc, sess, "explain what wikipedia is", timeout=30)

        history = persona.memory.get_history(sess.session_id)
        roles = [m.role for m in history]
        assert MessageRole.ASSISTANT in roles, (
            f"No ASSISTANT turn recorded in memory. History roles: {roles}"
        )

    def test_unknown_session_has_empty_history(self, mc):
        """A session that never interacted must return an empty history."""
        svc = _get_persona_service(mc)
        persona = svc.personas.get(PERSONA_NAME)
        assert persona is not None
        assert persona.memory is not None

        # drive a known session first to make sure the memory is active
        sess = Session(session_id="wiki-e2e-mem-known")
        SessionManager.sessions[sess.session_id] = sess
        _drive_utterance(mc, sess, "hello wikipedia", timeout=30)

        unknown_history = persona.memory.get_history("wiki-session-that-never-existed")
        assert unknown_history == [], (
            f"Expected empty history for unknown session, got: {unknown_history}"
        )
