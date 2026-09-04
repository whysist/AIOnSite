import pytest

from src.core.exceptions import LLMError
from src.llm.base import BaseLLM
from src.llm.echo_provider import EchoProvider
from src.llm.schemas import LLMRequest, LLMResponse, Message, Role, UsageMetadata


def test_request_from_loose_messages_normalises():
    req = LLMRequest.from_messages(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        model="m",
    )
    assert [m.role for m in req.messages] == [Role.SYSTEM, Role.USER]
    assert req.wire_messages()[0] == {"role": "system", "content": "s"}


def test_usage_addition():
    a = UsageMetadata(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    b = UsageMetadata(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert (a + b).total_tokens == 33


async def test_generate_returns_normalised_response():
    llm = EchoProvider()
    resp = await llm.generate([{"role": "user", "content": "hi"}])
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "echo"
    assert resp.latency_ms is not None
    assert resp.content


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 2}\n```', {"a": 2}),
        ('prose before {"a": 3} prose after', {"a": 3}),
    ],
)
def test_extract_json_tolerant(text, expected):
    assert BaseLLM.extract_json(text) == expected


def test_extract_json_raises_on_garbage():
    with pytest.raises(LLMError):
        BaseLLM.extract_json("no json here")


async def test_generate_json_via_echo_plan():
    llm = EchoProvider()
    obj = await llm.generate_json(
        [Message(role=Role.USER, content="do a thing")],
        metadata={"aionsite_kind": "plan"},
    )
    assert obj["goal"]
    assert len(obj["steps"]) >= 2


async def test_provider_errors_are_wrapped():
    class Boom(BaseLLM):
        provider_name = "boom"

        async def _complete(self, request):
            raise RuntimeError("kaboom")

    with pytest.raises(LLMError):
        await Boom().generate([{"role": "user", "content": "x"}])
