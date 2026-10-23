# Apply Module 2 to the existing Git-tracked Module 1 project

This build is designed to extend the already-pushed `v1.4-module-1-vennela` project.

The safest workflow is to keep your current repository folder (and therefore its `.git` history and local AGIQA-3K data) and overwrite only the Module 2 patch files.

After applying the patch:

```powershell
cd backend
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --reload
```

In another terminal:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Open `http://127.0.0.1:5173`.

The integrated build reuses `backend/data/runtime/module1_v14.db`, so Module 1 analyses already present in the clean v1.4 database remain valid. The database initialization only adds the Module 2 counterfactual table when needed.

Do not move or recreate the existing `v1.4-module-1-vennela` Git tag.
