from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.interfaces.api.schemas import ChatRequest


def test_chat_request_rejects_user_id_path_traversal():
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", user_id="../../etc/passwd")


def test_chat_request_rejects_user_id_dotdot():
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", user_id="..")


def test_chat_request_rejects_session_id_with_slash():
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", session_id="a/b")


def test_chat_request_accepts_long_message_up_to_32000_chars():
    request = ChatRequest(message="x" * 32000, user_id="user-1")
    assert len(request.message) == 32000
