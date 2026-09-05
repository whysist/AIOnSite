import asyncio

from src.api.streaming import ExecutionEventBus


def test_subscriber_sees_events_published_after_subscribing():
    bus = ExecutionEventBus()
    bus.create("e1")
    q = bus.subscribe("e1")
    bus.publish("e1", {"event_type": "node_started", "node": "a"})
    assert q.get_nowait() == {"event_type": "node_started", "node": "a"}


def test_late_subscriber_replays_buffered_history():
    bus = ExecutionEventBus()
    bus.create("e1")
    bus.publish("e1", {"event_type": "task_received"})
    bus.publish("e1", {"event_type": "plan_created"})
    q = bus.subscribe("e1")  # subscribes AFTER both events already happened
    assert q.get_nowait() == {"event_type": "task_received"}
    assert q.get_nowait() == {"event_type": "plan_created"}


def test_subscriber_after_finish_still_gets_terminal_event():
    bus = ExecutionEventBus()
    bus.create("e1")
    bus.publish("e1", {"event_type": "stream_end", "final_answer": "done"})
    q = bus.subscribe("e1")
    assert q.get_nowait()["event_type"] == "stream_end"


def test_events_are_scoped_per_execution():
    bus = ExecutionEventBus()
    bus.create("e1")
    bus.create("e2")
    q1 = bus.subscribe("e1")
    q2 = bus.subscribe("e2")
    bus.publish("e1", {"event_type": "x"})
    assert q1.get_nowait()["event_type"] == "x"
    assert q2.empty()


def test_unsubscribe_stops_further_delivery():
    bus = ExecutionEventBus()
    bus.create("e1")
    q = bus.subscribe("e1")
    bus.unsubscribe("e1", q)
    bus.publish("e1", {"event_type": "after_unsubscribe"})
    assert q.empty()


def test_buffer_is_capped():
    bus = ExecutionEventBus(max_buffer=5)
    bus.create("e1")
    for i in range(20):
        bus.publish("e1", {"event_type": "n", "i": i})
    q = bus.subscribe("e1")
    replayed = []
    while not q.empty():
        replayed.append(q.get_nowait())
    assert len(replayed) == 5
    assert replayed[-1]["i"] == 19
