"""Simple text statistics -- a pure, dependency-free tool."""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

from ..base_tool import BaseTool, ToolPermission

_WORD = re.compile(r"\b\w+\b")
_SENT = re.compile(r"[.!?]+")


class _In(BaseModel):
    text: str = Field(..., max_length=100_000)


class _Out(BaseModel):
    characters: int
    words: int
    sentences: int
    unique_words: int
    avg_word_length: float


class TextStatsTool(BaseTool[_In]):
    name = "text_stats"
    description = "Return character/word/sentence counts and vocabulary size for a piece of text."
    permissions: ClassVar = (ToolPermission.PURE,)
    InputModel = _In
    OutputModel = _Out

    async def _run(self, args: _In) -> _Out:
        words = _WORD.findall(args.text.lower())
        sentences = [s for s in _SENT.split(args.text) if s.strip()]
        avg_len = (sum(len(w) for w in words) / len(words)) if words else 0.0
        return _Out(
            characters=len(args.text),
            words=len(words),
            sentences=len(sentences),
            unique_words=len(set(words)),
            avg_word_length=round(avg_len, 3),
        )
