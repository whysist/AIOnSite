"""AnalyzeImageTool: path sandboxing (mirrors test_process_document.py); the
real Ollama wiring itself is covered by tests/test_vision/test_vlm.py.
"""

from __future__ import annotations

from src.tools.builtin import vision_tool
from src.tools.builtin.vision_tool import AnalyzeImageTool
from src.vision.schemas import VisionResult


async def test_path_traversal_is_blocked(tmp_path):
    secret = tmp_path.parent / "secret.png"
    secret.write_bytes(b"not a real image")
    tool = AnalyzeImageTool(root=tmp_path)
    result = await tool.run(file_path="../secret.png")
    assert result.ok is False
    assert "escapes" in result.error


async def test_missing_image_reports_execution_error(tmp_path):
    tool = AnalyzeImageTool(root=tmp_path)
    result = await tool.run(file_path="ghost.png")
    assert result.ok is False
    assert "image not found" in result.error


async def test_delegates_to_the_real_vlm_call(tmp_path, monkeypatch):
    image = tmp_path / "schematic.png"
    image.write_bytes(b"fake bytes")

    captured = {}

    async def fake_analyze(path, prompt="", **kwargs):
        captured["path"] = path
        captured["prompt"] = prompt
        return VisionResult(
            summary="V-101 schematic", detected_entities=["V-101"],
            confidence=0.9, raw_response={"model": "moondream:1.8b"},
        )

    monkeypatch.setattr(vision_tool, "analyze_image_with_vlm", fake_analyze)

    tool = AnalyzeImageTool(root=tmp_path)
    result = await tool.run(file_path="schematic.png", prompt="what tags are visible?")

    assert result.ok is True
    assert result.output["detected_entities"] == ["V-101"]
    assert result.output["model"] == "moondream:1.8b"
    assert captured["path"] == str(image.resolve())
    assert captured["prompt"] == "what tags are visible?"
