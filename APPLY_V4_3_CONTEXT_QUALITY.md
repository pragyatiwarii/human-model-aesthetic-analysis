# Apply v4.3 context-quality cleanup

Apply this patch on top of the working v4.2 context-enhanced project.

## What changes

- Module 1 context schema becomes `1.6-context-quality`.
- Detector objects are separated from prompt/VLM entities.
- Detector confidence is labelled high / medium / uncertain.
- Repeated VLM captions and duplicate self-relations are cleaned.
- Compound colour/view/detail attributes are normalised.
- Modules 2–4 remain backward compatible.

## Validate

From `backend` with the virtual environment active:

```powershell
python -m pytest -q
```

Expected: `30 passed`.

Then restart the backend and frontend and hard-refresh the browser.
