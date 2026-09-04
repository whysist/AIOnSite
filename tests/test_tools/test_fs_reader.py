
from src.tools.builtin.fs_reader import FileReadTool


async def test_reads_file_within_root(tmp_path):
    (tmp_path / "note.txt").write_text("hello sovereign world", encoding="utf-8")
    tool = FileReadTool(root=tmp_path)
    result = await tool.run(relative_path="note.txt")
    assert result.ok is True
    assert result.output["content"] == "hello sovereign world"


async def test_path_traversal_is_blocked(tmp_path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    tool = FileReadTool(root=tmp_path)
    result = await tool.run(relative_path="../secret.txt")
    assert result.ok is False
    assert "escapes" in result.error


async def test_missing_file_reports_error(tmp_path):
    tool = FileReadTool(root=tmp_path)
    result = await tool.run(relative_path="ghost.txt")
    assert result.ok is False
