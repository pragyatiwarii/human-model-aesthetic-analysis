# Apply Module 3 to the existing Git project

Use the **Module 3 patch ZIP** on top of the repository currently tagged `v2.0-module-2-sanyojita`.

Do not replace or delete:

- `.git/`
- `backend/data/agiqa3k/`
- `backend/data/runtime/`
- `backend/.venv/`
- `frontend/node_modules/`
- local `.env` files

Copy the contents of the patch into the project root and allow Windows to replace the listed source files.

Then run from `backend`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --reload
```

Open a second PowerShell in `frontend`:

```powershell
npm.cmd install
npm.cmd run dev
```

Open `http://127.0.0.1:5173`.

### First Module 3 test

1. Run Module 1 on `DALLE2_normal_000.jpg`.
2. In Module 2 generate Gaussian blur subtle/moderate/strong.
3. Scroll to **Module 3 - Muskan**.
4. Click **Train aesthetic model** once. Training is cached for later runs.
5. Click **Score original + 3 counterfactuals**.
6. Enter participant code `P001`.
7. Click **Open blind participant view**.
8. Give all four candidates a 1-5 rating and choose one Best candidate.
9. Submit the human preference.
10. Confirm candidate identities and model scores are revealed only after submission.
11. Confirm **Module 4 handoff ready** appears and opens successfully.

Do not push Module 3 until these checks pass locally.
