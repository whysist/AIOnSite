"""Real image / scanned-document understanding via the local VLM.

Wraps ``analyze_image_with_vlm`` (a genuine call to a local Ollama vision
model, see ``src/vision/vlm.py``) as an agent-callable tool, with the same
path-sandboxing pattern as ``FileReadTool``/``ProcessDocumentTool``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from ...core.exceptions import ToolExecutionError
from ...vision.vlm import DEFAULT_PROMPT, analyze_image_with_vlm
from ..base_tool import BaseTool, ToolPermission


class _AnalyzeImageIn(BaseModel):
    file_path: str = Field(..., description="Path to the image, relative to the data directory")
    prompt: str = Field(
        default=DEFAULT_PROMPT, max_length=1000,
        description="What to look for; the default asks for equipment tags and findings",
    )


class _AnalyzeImageOut(BaseModel):
    summary: str
    detected_entities: list[str]
    confidence: float
    model: str


class AnalyzeImageTool(BaseTool[_AnalyzeImageIn]):
    name = "analyze_image"
    description = (
        "Use the local vision-language model to describe an image, photo, "
        "schematic, or scanned page and identify equipment tags and findings "
        "in it. Use this for pictures and scans that text extraction can't "
        "read -- not for native-text PDFs (use process_document for those)."
    )
    # A loopback call to the already-trusted local Ollama server -- the same
    # trust boundary as the main LLM provider calls, which are never gated
    # by ToolPermission.NETWORK either. That permission exists to gate
    # arbitrary *external* network tools; marking this NETWORK would
    # silently drop it from default_registry() (constructed with
    # allow_network=False everywhere it's used -- see registry.py), making
    # it a tool no agent could ever actually call.
    permissions: ClassVar[tuple[ToolPermission, ...]] = (ToolPermission.READ_FILESYSTEM,)
    InputModel = _AnalyzeImageIn
    OutputModel = _AnalyzeImageOut

    def __init__(self, root: str | Path | None = None) -> None:
        self._root = Path(root or Path.cwd() / "data").resolve()

    async def _run(self, args: _AnalyzeImageIn) -> _AnalyzeImageOut:
        candidate = (self._root / args.file_path).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError:
            raise ToolExecutionError("path escapes the allowed data root") from None

        result = await analyze_image_with_vlm(str(candidate), prompt=args.prompt)
        return _AnalyzeImageOut(
            summary=result.summary,
            detected_entities=result.detected_entities,
            confidence=result.confidence,
            model=str(result.raw_response.get("model", "")),
        )
