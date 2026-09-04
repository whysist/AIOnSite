# Audit Layer

**Implemented.**  See `events.py` and `trail.py` (exported from
`src.audit`).

- `AuditEventType` — enumerated lifecycle events (`task_received`,
  `plan_created`, `pipeline_built`, `node_started/completed/failed/skipped`,
  `tool_called/failed`, `verification_completed`, `retry_triggered`,
  `replan_triggered`, `final_answer_generated`, `execution_completed/failed`).
- `AuditEvent` — immutable (`frozen`) record: `timestamp`, `execution_id`,
  `event_type`, `component`, `status`, `message`, sanitised `metadata`.
- `AuditTrail` — append-only in-memory log with secret redaction and an
  optional `on_event` sink for durable storage. Query per execution with
  `events(execution_id)` / `as_dicts(execution_id)`.

Exposed over HTTP at `GET /tasks/{execution_id}/audit`.
