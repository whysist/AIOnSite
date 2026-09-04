# AIOnSite frontend

A small, dependency-free static page that drives the real backend API --
no build step, no framework, no bundler. It talks directly to the FastAPI
app in `src/api/app.py`: every number and piece of text on the page comes
from a real `/health`, `/tasks`, `/tasks/{id}`, `/tasks/{id}/pipeline` or
`/tasks/{id}/audit` response.

```
frontend/
  index.html   page structure
  styles.css   look & feel
  app.js       all behaviour (fetch calls, rendering, local history)
```

## Run it

1. Start the backend (from the project root):

   ```bash
   python -m src.main --serve --port 8000
   ```

2. Open `frontend/index.html` directly in a browser (double-click it, or
   drag it into a tab) -- or serve the folder with any static server:

   ```bash
   cd frontend
   python -m http.server 5500
   # then open http://localhost:5500
   ```

   Both work. The backend has CORS enabled (`allow_origins=["*"]`,
   `src/api/app.py`), so it doesn't matter whether the page is opened as a
   `file://` URL or served from a different port.

3. If your backend isn't on `http://localhost:8000`, change the **API base
   URL** field in the sidebar -- it's saved in the browser (`localStorage`)
   so you only need to set it once.

## What it does

- **Sidebar**: live backend health (environment, provider, sovereign mode,
  registered tool count), and a run history you can click back into.
- **New task**: submit any task text (optionally `confidential`) to
  `POST /tasks`. This call is synchronous on the backend today, so the page
  shows an honest "Running... Ns" state with an elapsed timer -- it does
  not fabricate per-node progress while the request is in flight.
- **Result**: execution status, task status, duration, the final answer
  (flagged as *provisional* when `final_answer_degraded` is true), and the
  full verification verdict (score, recommendation, issues, missing
  requirements).
- **Pipeline**: every DAG node -- type, criticality, outcome, provider/model,
  dependencies, attempts. Click a node to expand it and see its raw output
  plus every tool call it made (arguments, result, error, duration).
- **Audit trail**: the complete ordered event log for the run.

## Local history vs. server truth

`GET /tasks` only returns bare execution ids (the backend doesn't store
task text against them), so the page keeps a small `localStorage` map of
`id -> task text` purely for readable labels in the sidebar. The backend's
own execution store is always the source of truth for status, results,
pipeline and audit data; the local history is a label cache only, and
survives a browser refresh but not a different browser/machine.

## Known limitation

`POST /tasks` blocks until the run finishes (see the backend docs, "known
limitations" -- a background-job/streaming variant is a listed follow-up).
For a real local model this can take minutes; the offline `echo` provider
(`LLM_PROVIDER=echo`) returns near-instantly and is the fastest way to
exercise this page end-to-end.
