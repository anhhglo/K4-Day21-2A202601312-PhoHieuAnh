# Income Model MLOps CI/CD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automate the Adult Income model lifecycle from deterministic local data preparation through MLflow, DVC, a gated four-job GitHub Actions pipeline, and deployment to a live GCP VM.

**Architecture:** Local scripts prepare and DVC-version deterministic datasets, while `src/train.py` produces an MLflow run, report, and serialized model. GitHub Actions transfers the candidate model between jobs as an artifact, promotes it to GCS only after the positive-class F1 gate passes, then updates and restarts a FastAPI service on Compute Engine.

**Tech Stack:** Python 3.10, pandas, scikit-learn, MLflow, DVC with Google Cloud Storage, FastAPI, pytest, GitHub Actions, Google Cloud Storage, Compute Engine, systemd.

**Spec:** `docs/superpowers/specs/2026-08-21-income-mlops-cicd-design.md`

## Global Constraints

- Work only in `/home/anh2/VinUniCo/DAY21` except for authenticated cloud and GitHub operations explicitly named in this plan.
- Use GCP project `track2-day16-01312`, region `us-central1`, and zone `us-central1-a`.
- Prefer bucket `k4-day21-2a202601312-phohieuanh`; use `k4-day21-2a202601312-phohieuanh-01312` only when the preferred name is unavailable.
- Use VM `income-api`, machine type `e2-small`, and service account `income-lab-sa`.
- Use Python 3.10 or newer and `GradientBoostingClassifier(random_state=42)`.
- Gate deployment on positive-class `f1_score >= 0.65`; accuracy is never a deployment gate.
- Never track CSV data, model binaries, SQLite databases, MLflow artifacts, `.dvc/config.local`, cloud keys, or SSH private keys.
- Push DVC objects before pushing a Git commit that references them.
- Promote `artifacts/current/model.joblib` only in the Release job after Quality Gate succeeds.
- Do not delete GCP resources without a separate explicit user confirmation.

---

### Task 1: Bootstrap the Python project and secret-safe repository

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `params.yaml`
- Create: `src/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md`
- Modify: none

**Interfaces:**
- Consumes: the approved design document.
- Produces: Python virtual environment `.venv`, importable `src` package, dependency manifest, default hyperparameters, and ignore rules used by every later task.

- [ ] **Step 1: Add a failing repository-safety test**

Create `tests/test_repository_safety.py`:

```python
from pathlib import Path


REQUIRED_IGNORES = {
    ".venv/",
    ".secrets/",
    ".dvc/config.local",
    "data/*.csv",
    "models/",
    "outputs/",
    "mlflow.db",
    "mlartifacts/",
    "sa-key.json",
    "*.pem",
    "*.key",
}


def test_sensitive_and_generated_files_are_ignored():
    rules = {
        line.strip()
        for line in Path(".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert REQUIRED_IGNORES <= rules
```

- [ ] **Step 2: Create the test environment, run the safety test, and confirm the expected failure**

Run: `python3 -m venv .venv`

Run: `.venv/bin/python -m pip install pytest`

Run: `.venv/bin/python -m pytest tests/test_repository_safety.py -v`

Expected before bootstrap: FAIL because `.gitignore` does not exist.

- [ ] **Step 3: Create the environment and manifests**

Create `requirements.txt` with bounded dependencies:

```text
dvc[gs]>=3.50,<4
fastapi>=0.110,<1
google-cloud-storage>=2.16,<4
httpx>=0.27,<1
joblib>=1.3,<2
mlflow>=2.12,<4
numpy>=1.26,<3
pandas>=2.2,<3
pydantic>=2.6,<3
pytest>=8,<10
PyYAML>=6,<7
scikit-learn>=1.4,<2
uvicorn>=0.29,<1
```

Create `.gitignore` with the exact required rules plus Python caches:

```gitignore
.venv/
.secrets/
.env
.dvc/config.local
data/*.csv
models/
outputs/
mlflow.db
mlartifacts/
sa-key.json
*.pem
*.key
__pycache__/
*.py[cod]
.pytest_cache/
mlruns/
```

Create `params.yaml`:

```yaml
n_estimators: 100
learning_rate: 0.1
max_depth: 3
```

Create empty package markers and a `README.md` that states the project purpose,
Python requirement, F1 threshold, and the four pipeline jobs.

- [ ] **Step 4: Install dependencies and rerun the safety test**

Run: `.venv/bin/python -m pip install --upgrade pip`

Run: `.venv/bin/python -m pip install -r requirements.txt`

Run: `.venv/bin/python -m pytest tests/test_repository_safety.py -v`

Expected: PASS.

- [ ] **Step 5: Confirm no secret-like files are staged and commit**

Run: `git status --short`

Run: `git check-ignore .secrets/private.key data/example.csv models/model.joblib outputs/report.json sa-key.json`

Expected: every probe path is ignored.

Commit:

```bash
git add .gitignore requirements.txt params.yaml README.md src/__init__.py tests/__init__.py tests/test_repository_safety.py
git commit -m "chore: bootstrap Day 21 MLOps lab"
```

---

### Task 2: Prepare and append deterministic Adult Census data

**Files:**
- Create: `prepare_data.py`
- Create: `append_batch.py`
- Create: `tests/test_prepare_data.py`
- Test: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: UCI `adult.data` and `adult.test` tabular formats.
- Produces: `prepare_data.prepare_frames(raw_train, raw_test) -> pandas.DataFrame`, `prepare_data.split_frames(cleaned) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]`, and three CSV files with the canonical eleven-column schema.

- [ ] **Step 1: Write failing transformation and split tests**

Test exact categorical ordering, label normalization, missing-row removal, and a
deterministic split helper. The core assertions are:

```python
def test_prepare_frames_encodes_categories_alphabetically():
    cleaned = prepare_frames(_raw_train_fixture(), _raw_test_fixture())
    assert list(cleaned.columns) == FEATURE_NAMES + ["target"]
    assert set(cleaned["target"]) == {0, 1}
    assert cleaned["workclass"].tolist() == [1, 0, 1]


def test_split_frames_uses_fixed_holdout_and_equal_batches():
    frame = _canonical_frame(10)
    batch1, holdout, batch2 = split_frames(
        frame, holdout_size=2, random_state=42
    )
    assert (len(batch1), len(holdout), len(batch2)) == (4, 2, 4)
    assert set(batch1.index).isdisjoint(holdout.index)
    assert set(batch2.index).isdisjoint(holdout.index)
```

Add an append test that writes four-row batch files, runs
`append_batch.append_batch`, verifies eight rows, and verifies a second call
raises `ValueError("train_batch1 already contains batch2")`.

- [ ] **Step 2: Run the data tests and confirm they fail**

Run: `.venv/bin/python -m pytest tests/test_prepare_data.py -v`

Expected: FAIL with import errors for `prepare_data` and `append_batch`.

- [ ] **Step 3: Implement deterministic cleaning and splitting**

Define the complete raw UCI column list, canonical feature list, and URLs:

```python
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
FEATURE_NAMES = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]
```

Implementation rules:

```python
combined = pd.concat([raw_train, raw_test], ignore_index=True)
combined = combined.replace(r"^\s*\?\s*$", np.nan, regex=True).dropna()
combined["target"] = (
    combined["target"].astype(str).str.strip().str.rstrip(".").eq(">50K").astype(int)
)
for column in ["workclass", "marital_status", "occupation", "relationship", "sex"]:
    values = sorted(combined[column].astype(str).str.strip().unique())
    combined[column] = combined[column].astype(str).str.strip().map(
        {value: index for index, value in enumerate(values)}
    )
```

`split_frames` must use `frame.sample(frac=1, random_state=42)`, reserve the
first 500 shuffled rows for holdout in the production call, then divide the
remaining 44,722 rows equally. `main()` requires 45,222 cleaned rows before
writing the three CSV files.

Implement `append_batch(path1, path2)` with schema and exact pre/post row-count
checks before an atomic CSV replacement.

- [ ] **Step 4: Run tests, download real data, and verify exact outputs**

Run: `.venv/bin/python -m pytest tests/test_prepare_data.py -v`

Expected: PASS.

Run: `.venv/bin/python prepare_data.py`

Expected output includes `22361`, `500`, `22361`, and a positive-class rate near
`24.8%`.

Run:

```bash
.venv/bin/python -c "import pandas as pd; print([len(pd.read_csv(p)) for p in ['data/train_batch1.csv','data/holdout.csv','data/train_batch2.csv']])"
```

Expected: `[22361, 500, 22361]`.

- [ ] **Step 5: Verify CSVs are ignored and commit only code/tests**

Run: `git status --short --ignored data`

Expected: the CSV files appear ignored, not untracked.

Commit:

```bash
git add prepare_data.py append_batch.py tests/test_prepare_data.py
git commit -m "feat: prepare deterministic Adult Income datasets"
```

---

### Task 3: Implement training, validation, MLflow logging, and artifacts

**Files:**
- Create: `src/train.py`
- Create: `tests/test_train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `params: dict`, a training CSV, and a held-out evaluation CSV with matching features and a binary `target`.
- Produces: `train(params: dict, data_path: str = "data/train_batch1.csv", eval_path: str = "data/holdout.csv") -> float`, `outputs/report.json`, and `models/model.joblib`.

- [ ] **Step 1: Write the failing training tests**

Use `tmp_path` and `monkeypatch.chdir(tmp_path)` so artifacts never leak into the
repository. Create deterministic 200-row synthetic data and assert:

```python
f1 = train(
    {"n_estimators": 10, "learning_rate": 0.1, "max_depth": 2},
    data_path=str(train_path),
    eval_path=str(eval_path),
)
assert isinstance(f1, float)
assert 0.0 <= f1 <= 1.0
report = json.loads(Path("outputs/report.json").read_text())
assert set(report) == {"f1_score", "accuracy"}
assert all(isinstance(value, float) for value in report.values())
assert joblib.load("models/model.joblib").n_features_in_ == 10
```

Also test missing `target`, mismatched features, and a one-class training target;
each must raise `ValueError` with a specific explanatory message.

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`

Expected: FAIL because `src.train` does not exist.

- [ ] **Step 3: Implement the minimal validated training flow**

Use these metric calls and fixed model seed:

```python
model = GradientBoostingClassifier(**params, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_eval)
f1 = float(f1_score(y_eval, predictions))
accuracy = float(accuracy_score(y_eval, predictions))
```

Create or select experiment `adult-income`; when creating it, set
`artifact_location=Path("mlartifacts").resolve().as_uri()`. Within
`mlflow.start_run()`, call `mlflow.log_params(params)`, log metrics named exactly
`f1_score` and `accuracy`, and call `mlflow.sklearn.log_model(model, "model")`.
Create the output directories, serialize JSON with UTF-8 and a final newline,
dump the model with joblib, print both metrics, and return `f1`.

The CLI entry point reads `params.yaml` with `yaml.safe_load` and calls `train`.
Honor `MLFLOW_TRACKING_URI`; default it to `sqlite:///mlflow.db` when absent.

- [ ] **Step 4: Run targeted and complete local tests**

Run: `.venv/bin/python -m pytest tests/test_train.py -v`

Expected: all training tests PASS.

Run: `.venv/bin/python -m pytest tests -v`

Expected: all current tests PASS.

- [ ] **Step 5: Commit training code and tests**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: train and track the income model"
```

---

### Task 4: Implement the testable FastAPI inference service

**Files:**
- Create: `src/serve.py`
- Create: `tests/test_serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `ARTIFACT_BUCKET`, optional `MODEL_PATH`, GCS object `artifacts/current/model.joblib`, and a request containing ten numeric features.
- Produces: `download_model() -> pathlib.Path`, `create_app(model_loader: Callable) -> FastAPI`, `GET /healthz`, and `POST /score`.

- [ ] **Step 1: Write failing endpoint tests with an injected fake model**

Use a fake model whose `predict` returns a configured integer:

```python
class FakeModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, rows):
        assert len(rows[0]) == 10
        return np.array([self.prediction])


def test_health_and_low_income_score():
    app = create_app(lambda: FakeModel(0))
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        response = client.post("/score", json={"features": [0.0] * 10})
        assert response.json() == {"prediction": 0, "label": "thu_nhap_thap"}
```

Add corresponding high-income and nine-feature HTTP 400 assertions.

- [ ] **Step 2: Run serving tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_serve.py -v`

Expected: FAIL because `create_app` does not exist.

- [ ] **Step 3: Implement startup loading and endpoints**

Use a FastAPI lifespan so importing the module does not contact GCS during test
collection:

```python
def create_app(model_loader=load_runtime_model):
    @asynccontextmanager
    async def lifespan(app):
        app.state.model = model_loader()
        yield

    api = FastAPI(lifespan=lifespan)
```

`download_model` creates the parent directory, downloads to a sibling `.tmp`
file, calls `os.replace(temp_path, model_path)`, and returns the final path.
`load_runtime_model` calls `joblib.load(download_model())`.

The score handler checks `len(req.features) == 10`, converts the prediction to
`int`, and maps labels with `{0: "thu_nhap_thap", 1: "thu_nhap_cao"}`.

- [ ] **Step 4: Run service and full tests**

Run: `.venv/bin/python -m pytest tests/test_serve.py -v`

Run: `.venv/bin/python -m pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit serving code and tests**

```bash
git add src/serve.py tests/test_serve.py
git commit -m "feat: serve income predictions with FastAPI"
```

---

### Task 5: Run MLflow experiments and select production parameters

**Files:**
- Modify: `params.yaml`
- Create: `REPORT.md`
- Generated, ignored: `mlflow.db`
- Generated, ignored: `mlartifacts/`
- Generated, ignored: `outputs/report.json`
- Generated, ignored: `models/model.joblib`

**Interfaces:**
- Consumes: real batch 1 and holdout CSVs plus the three parameter sets in the design.
- Produces: at least three MLflow runs, measured metric table, and the best passing parameter set in `params.yaml`.

- [ ] **Step 1: Run the baseline experiment**

Write baseline parameters to `params.yaml`, then run:

`MLFLOW_TRACKING_URI=sqlite:///mlflow.db .venv/bin/python src/train.py`

Record the printed F1 and accuracy immediately in the REPORT experiment table.

- [ ] **Step 2: Run the weak comparison experiment**

Set `50`, `0.05`, and `2`; rerun the same command and record both metrics.

- [ ] **Step 3: Run the strong candidate experiment**

Set `200`, `0.1`, and `5`; rerun and record both metrics.

- [ ] **Step 4: Query MLflow programmatically and choose the best passing run**

Run:

```bash
.venv/bin/python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); print(mlflow.search_runs()[['params.n_estimators','params.learning_rate','params.max_depth','metrics.f1_score','metrics.accuracy']].sort_values('metrics.f1_score', ascending=False).to_string(index=False))"
```

Expected: at least three rows. Select the highest F1 row at or above 0.65 and
write those exact parameters to `params.yaml`.

- [ ] **Step 5: Complete the first REPORT section and commit**

`REPORT.md` must include the real three-row metric table, chosen parameters,
and this evidence-based explanation: the approximately 75/25 imbalance permits
a majority-only accuracy near 0.752 while positive-class F1 is 0, so F1 protects
the minority outcome that the service must detect.

Commit only tracked configuration and report text:

```bash
git add params.yaml REPORT.md
git commit -m "docs: record MLflow experiment results"
```

---

### Task 6: Configure DVC and provision bucket identity

**Files:**
- Create: `.dvc/config`
- Create: `data/train_batch1.csv.dvc`
- Create: `data/holdout.csv.dvc`
- Create: `data/train_batch2.csv.dvc`
- Generated, ignored: `.dvc/config.local`
- Generated, ignored: `.secrets/sa-key.json`

**Interfaces:**
- Consumes: three prepared CSVs and authenticated gcloud CLI.
- Produces: GCS DVC remote, service-account JSON for GitHub only, DVC pointers, and uploaded DVC objects.

- [ ] **Step 1: Enable required APIs and resolve the bucket name**

Run `gcloud services enable storage.googleapis.com compute.googleapis.com iam.googleapis.com --project track2-day16-01312`.

Check the preferred bucket with `gcloud storage buckets describe
gs://k4-day21-2a202601312-phohieuanh`. If it exists in this project, use it. If
it does not exist, create it in `us-central1`; if GCS reports global-name
ownership by another project, create the documented `-01312` fallback instead.
Persist the selected non-secret name in the shell variable `LAB_BUCKET_NAME`
for this task and in the GitHub secret during Task 9.

- [ ] **Step 2: Create or verify the service account and bucket IAM**

Create `income-lab-sa` only when it does not exist. Grant
`roles/storage.objectAdmin` on only the selected bucket, not the whole project.
Do not grant `roles/storage.admin`.

- [ ] **Step 3: Create the ignored key file and verify its permissions**

Create `.secrets/`, set mode 700, create `.secrets/sa-key.json` for
`income-lab-sa`, and set mode 600. If a valid key already exists, reuse it
instead of creating another service-account key.

Run: `git check-ignore .secrets/sa-key.json`

Expected: `.secrets/sa-key.json` is ignored.

- [ ] **Step 4: Initialize DVC and push the three datasets**

Run:

```bash
.venv/bin/dvc init
.venv/bin/dvc remote add -d labstore "gs://${LAB_BUCKET_NAME}/dvc"
.venv/bin/dvc remote modify --local labstore credentialpath .secrets/sa-key.json
.venv/bin/dvc add data/train_batch1.csv data/holdout.csv data/train_batch2.csv
.venv/bin/dvc push
```

Verify with `.venv/bin/dvc status -c` and `gcloud storage ls --recursive
"gs://${LAB_BUCKET_NAME}/dvc/**"`.

- [ ] **Step 5: Prove only DVC pointers are staged and commit**

Run: `git status --short`

Run: `git ls-files data`

Expected: only three `.dvc` pointer paths; no CSV.

Commit:

```bash
git add .dvc/config .dvc/.gitignore data/.gitignore data/train_batch1.csv.dvc data/holdout.csv.dvc data/train_batch2.csv.dvc
git commit -m "feat: version Adult datasets with DVC"
```

---

### Task 7: Define the gated workflow and reproducible systemd service

**Files:**
- Create: `.github/workflows/cicd.yml`
- Create: `deploy/income-api.service`
- Create: `tests/test_delivery_config.py`
- Test: `tests/test_delivery_config.py`

**Interfaces:**
- Consumes: GitHub secrets, DVC remote, trained model/report, and VM SSH access.
- Produces: four-job workflow with safe promotion and systemd unit named `income-api`.

- [ ] **Step 1: Write failing static delivery tests**

Parse YAML using `yaml.BaseLoader` and assert:

```python
jobs = workflow["jobs"]
assert list(jobs) == ["unit-test", "train", "quality-gate", "release"]
assert jobs["train"]["needs"] == "unit-test"
assert jobs["quality-gate"]["needs"] == "train"
assert jobs["release"]["needs"] == "quality-gate"
assert "float(" in jobs["quality-gate"]["steps"][0]["run"]
assert "artifacts/current/model.joblib" not in str(jobs["train"])
assert "artifacts/current/model.joblib" in str(jobs["release"])
```

Read the service unit and assert `Restart=always`, `RestartSec=5`, port 8080,
and `EnvironmentFile=/etc/income-api.env`.

- [ ] **Step 2: Run delivery tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/test_delivery_config.py -v`

Expected: FAIL because workflow and systemd files do not exist.

- [ ] **Step 3: Implement the four-job workflow**

Trigger on manual dispatch and pushes to `main` that change `data/*.dvc`,
`src/**/*.py`, `tests/**/*.py`, `prepare_data.py`, `append_batch.py`,
`params.yaml`, `requirements.txt`, or `.github/workflows/cicd.yml`. Use official
checkout/setup/auth/artifact actions. The Train job sets F1 with:

```bash
F1=$(python -c "import json; print(json.load(open('outputs/report.json'))['f1_score'])")
echo "f1=${F1}" >> "$GITHUB_OUTPUT"
```

The gate uses:

```python
f1 = float("${{ needs.train.outputs.f1 }}")
if f1 < 0.65:
    raise SystemExit(f"FAILED: f1_score {f1:.4f} < 0.65")
print(f"PASSED: f1_score {f1:.4f} >= 0.65")
```

Train uploads `models/model.joblib` and `outputs/report.json` as artifact
`candidate-model`. Release downloads that artifact, authenticates to GCP,
uploads only then to the bucket secret, copies `src/serve.py`,
`requirements.txt`, and the systemd unit to the VM, renders the documented
`__SERVER_USER__` substitution token from the `SERVER_USER` secret, installs
runtime dependencies under that user's `income-api/.venv`, installs the rendered
service unit and environment file with sudo, restarts the service, and retries
`/healthz` for up to 60 seconds.

- [ ] **Step 4: Implement and verify the systemd unit**

The unit uses:

```ini
[Unit]
Description=Income Model Inference Server
After=network-online.target
Wants=network-online.target

[Service]
User=__SERVER_USER__
WorkingDirectory=/home/__SERVER_USER__/income-api
EnvironmentFile=/etc/income-api.env
ExecStart=/home/__SERVER_USER__/income-api/.venv/bin/uvicorn src.serve:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Run: `.venv/bin/python -m pytest tests/test_delivery_config.py -v`

Run: `.venv/bin/python -m pytest tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Validate formatting and commit**

Run: `git diff --check`

Commit:

```bash
git add .github/workflows/cicd.yml deploy/income-api.service tests/test_delivery_config.py
git commit -m "feat: automate gated model delivery"
```

---

### Task 8: Provision and bootstrap the Compute Engine VM

**Files:**
- Generated, ignored: `.secrets/income_deploy`
- Modify: `README.md`

**Interfaces:**
- Consumes: selected bucket, service account, repository deployment files, and gcloud authentication.
- Produces: VM public IP, SSH deployment identity, open TCP 8080 firewall rule, and a runtime prepared for systemd deployment.

- [ ] **Step 1: Create or verify the firewall rule and VM**

Create firewall rule `allow-income-api` only if absent, allowing TCP 8080 to
target tag `income-api`. Create VM `income-api` only if absent with Ubuntu 22.04,
`e2-small`, tag `income-api`, the `income-lab-sa` identity, and
`cloud-platform` scope.

- [ ] **Step 2: Generate a repository-local ignored deployment key**

Run `ssh-keygen -t ed25519 -f .secrets/income_deploy -N "" -C
"github-actions-deploy"` only if the key does not exist. Set private-key mode
600 and confirm `git check-ignore .secrets/income_deploy` succeeds.

- [ ] **Step 3: Authorize the public key without printing private material**

Use `gcloud compute ssh income-api --zone us-central1-a` to append the contents
of `.secrets/income_deploy.pub` to the VM user's `~/.ssh/authorized_keys` only
when the exact public key is not already present. Never read or echo the private
key to terminal output.

- [ ] **Step 4: Discover the VM user, prepare runtime directories, and verify metadata credentials**

Run `gcloud compute ssh income-api --zone us-central1-a --command whoami` and
store the exact non-secret result as `LAB_SERVER_USER`. Over gcloud SSH, create
`/home/${LAB_SERVER_USER}/income-api/src`, install `python3-venv`, create the
runtime venv, and install `google-cloud-storage`. Confirm a short Python
`google.cloud.storage.Client()` probe resolves the attached service account and
can read the selected bucket without a JSON key on the VM.

- [ ] **Step 5: Record safe connection metadata and commit documentation**

Add project, zone, VM name, service name, and non-secret verification commands
to README. Do not record private-key contents or service-account JSON.

Commit:

```bash
git add README.md
git commit -m "docs: add VM deployment operations"
```

---

### Task 9: Configure GitHub secrets and run the first deployment

**Files:**
- Modify only if diagnosed: workflow/source files associated with a concrete failure.
- External state: five GitHub Actions repository secrets and the `main` branch.

**Interfaces:**
- Consumes: service-account JSON, bucket name, VM IP/user, deployment private key, completed local commits.
- Produces: first successful four-job GitHub Actions run and live production endpoint.

- [ ] **Step 1: Run the pre-push verification gate**

Run:

```bash
.venv/bin/python -m pytest tests -v
git diff --check
git status --short
if git ls-files | rg -q '(^|/)(sa-key\.json|.*\.csv|model\.joblib|mlflow\.db|income_deploy)$'; then echo "tracked generated or secret file detected"; exit 1; fi
```

Expected: tests PASS, no whitespace errors, and no sensitive/generated matches.

- [ ] **Step 2: Set exactly five repository secrets**

Use `gh secret set` with stdin for `STORAGE_CREDENTIALS` and `SERVER_SSH_KEY`.
Set `ARTIFACT_BUCKET`, the VM external IP as `SERVER_HOST`, and the discovered
`LAB_SERVER_USER` as `SERVER_USER`. List secret names with `gh secret list`; do
not print values.

- [ ] **Step 3: Push main and watch the workflow**

Run: `git push -u origin main`

Run: `gh run list --workflow cicd.yml --limit 3`

Run: `gh run watch --exit-status`

Expected: Unit Test, Train, Quality Gate, and Release all succeed.

- [ ] **Step 4: Diagnose any failed job systematically and rerun**

Read only failed logs with `gh run view --log-failed`. Fix the root cause in a
focused test-first commit, rerun the complete local test suite, push, and watch
the replacement run. Do not weaken the F1 threshold or bypass a failed health
check.

- [ ] **Step 5: Verify cloud artifact and live endpoints**

Require all commands to succeed:

```bash
gcloud storage ls "gs://${LAB_BUCKET_NAME}/artifacts/current/model.joblib"
curl -sf "http://${SERVER_HOST}:8080/healthz"
curl -sf -X POST "http://${SERVER_HOST}:8080/score" -H "Content-Type: application/json" -d '{"features":[60,2,5,2,4,0,1,0,0,45]}'
curl -sf -X POST "http://${SERVER_HOST}:8080/score" -H "Content-Type: application/json" -d '{"features":[28,2,14,2,11,0,1,0,0,45]}'
```

Record the run URL and returned predictions in REPORT without recording secrets.

---

### Task 10: Demonstrate that the quality gate blocks release

**Files:**
- Modify temporarily and restore: `params.yaml`
- Modify: `REPORT.md`

**Interfaces:**
- Consumes: functioning workflow and production model.
- Produces: GitHub evidence of a failed Quality Gate with skipped Release while the prior production artifact remains in place.

- [ ] **Step 1: Record the current production object's generation/hash**

Use `gcloud storage objects describe` on
`artifacts/current/model.joblib` and record a non-secret generation or MD5 value.

- [ ] **Step 2: Commit a deliberately non-learning configuration**

Set:

```yaml
n_estimators: 1
learning_rate: 0.01
max_depth: 1
```

Run local training to confirm positive-class F1 is below 0.65, then commit and
push with message `test: demonstrate F1 quality gate`.

- [ ] **Step 3: Verify gate failure and skipped release**

Watch the GitHub run. Require Unit Test and Train to pass, Quality Gate to fail,
and Release to be skipped. Verify the production object's generation/hash is
unchanged and `/healthz` still returns OK.

- [ ] **Step 4: Restore the selected production parameters**

Restore the exact best parameters recorded in REPORT, run local training and
tests, commit `fix: restore production model parameters`, push, and require all
four jobs to pass.

- [ ] **Step 5: Record the negative-test run URL and outcome**

Add the weak local F1, failed run URL, skipped Release status, and unchanged
artifact evidence to REPORT; commit with `docs: record quality gate evidence`.

---

### Task 11: Prove continuous training from a data-only commit

**Files:**
- Modify generated, ignored: `data/train_batch1.csv`
- Modify tracked: `data/train_batch1.csv.dvc`
- Modify after evidence: `REPORT.md`

**Interfaces:**
- Consumes: untouched batch 2, selected production parameters, DVC remote, and successful initial pipeline.
- Produces: 44,722-row batch 1 pointer, DVC object uploaded before Git push, successful data-triggered workflow, and before/after F1 comparison.

- [ ] **Step 1: Append once and verify exact row count**

Run: `.venv/bin/python append_batch.py`

Run: `.venv/bin/python -c "import pandas as pd; assert len(pd.read_csv('data/train_batch1.csv')) == 44722"`

Expected: both commands succeed.

- [ ] **Step 2: Update the DVC pointer and prove CSV remains ignored**

Run: `.venv/bin/dvc add data/train_batch1.csv`

Run: `git status --short`

Expected: only `data/train_batch1.csv.dvc` is modified for data.

- [ ] **Step 3: Push DVC before committing/pushing Git**

Run: `.venv/bin/dvc push`

Run: `.venv/bin/dvc status -c`

Expected: cloud and workspace are synchronized.

- [ ] **Step 4: Create and push the data-only commit**

```bash
git add data/train_batch1.csv.dvc
git commit -m "data: bổ sung 22361 mẫu dữ liệu mới"
git show --name-only --format= HEAD
git push origin main
```

Expected `git show`: only `data/train_batch1.csv.dvc`.

- [ ] **Step 5: Verify automated retraining and record metric comparison**

Watch the triggered GitHub run and require all four jobs to pass. Download its
`report` artifact, compare its F1 with the 22,361-row production run, and add the
real values plus same-distribution explanation to REPORT. Commit and push the
report update only after capturing the data-triggered run URL.

---

### Task 12: Final verification and submission handoff

**Files:**
- Modify: `README.md`
- Modify: `REPORT.md`
- Create: `submission/CHECKLIST.md`
- Test: complete repository and external system checks.

**Interfaces:**
- Consumes: final repository, MLflow database, successful workflow runs, GCS objects, and live VM.
- Produces: submission-ready report, reproducible commands, evidence checklist, and cleanup instructions.

- [ ] **Step 1: Run fresh local verification**

Run:

```bash
.venv/bin/python -m pytest tests -v
git diff --check
.venv/bin/dvc status
.venv/bin/dvc status -c
git status --short
```

Record exact test counts and confirm no unexpected changes.

- [ ] **Step 2: Run fresh external verification**

List the two successful workflow runs and negative gate run, list GCS DVC and
model objects, call health and score endpoints, and inspect
`gcloud compute instances describe income-api` for RUNNING status and external
IP. Store only non-secret results in the report/checklist.

- [ ] **Step 3: Finish the one-page report**

Ensure REPORT includes selected parameters and rationale, real three-run MLflow
metrics, positive-class F1 explanation, 22,361-versus-44,722 F1 comparison,
quality-gate evidence, and difficulties/resolutions. Keep the main narrative to
approximately one A4 page.

- [ ] **Step 4: Create the ordered screenshot checklist**

`submission/CHECKLIST.md` lists:

1. MLflow comparison with parameters, `f1_score`, and accuracy visible.
2. Initial four-green-job GitHub run.
3. Data-commit four-green-job GitHub run.
4. Health and both score curl outputs.
5. GCS `dvc/` objects and `artifacts/current/model.joblib`.
6. Failed Quality Gate with skipped Release.

Include the exact run URLs, VM IP, bucket path, and commands needed to open each
view, but no credentials.

- [ ] **Step 5: Commit and push final documentation**

```bash
git add README.md REPORT.md submission/CHECKLIST.md
git commit -m "docs: finalize Day 21 submission evidence"
git push origin main
```

Watch the documentation-trigger behavior: documentation-only changes should not
start retraining. Confirm the repository is clean and public. Provide teardown
commands in README, but do not execute them.
