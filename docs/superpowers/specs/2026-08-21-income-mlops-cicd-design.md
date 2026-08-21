# Income Model MLOps CI/CD Design

## 1. Purpose

Build the complete Day 21 K4 lab in the public repository
`anhhglo/K4-Day21-2A202601312-PhoHieuAnh`. A data-pointer commit must trigger
testing, model training, an F1 quality gate, safe model promotion, deployment to
a GCP VM, and an automated health check.

The final system must provide four observable outcomes:

1. MLflow contains at least three comparable runs with `f1_score`, `accuracy`,
   and distinct hyperparameters.
2. GitHub Actions shows four successful jobs in order: Unit Test, Train,
   Quality Gate, and Release.
3. A public VM answers `GET /healthz` and `POST /score` on port 8080.
4. Google Cloud Storage contains DVC objects below `dvc/` and the deployed
   model at `artifacts/current/model.joblib`.

## 2. Environment and Cloud Scope

- Local environment: WSL2, Ubuntu 22.04.5 LTS, Python 3.10.12.
- GCP project: `track2-day16-01312`.
- GCP region and zone: `us-central1` and `us-central1-a`.
- Preferred bucket: `k4-day21-2a202601312-phohieuanh`.
- Bucket fallback if the preferred globally unique name is unavailable:
  `k4-day21-2a202601312-phohieuanh-01312`.
- VM: `income-api`, Ubuntu 22.04 LTS, machine type `e2-small`.
- Training/deployment service account: `income-lab-sa`.
- Runtime port: TCP 8080.

Creating the bucket, VM, service account, firewall rule, and stored objects can
incur GCP charges. Resources are limited to the named project and lab scope.

## 3. Repository Layout

```text
.
├── .dvc/config
├── .github/workflows/cicd.yml
├── .gitignore
├── README.md
├── REPORT.md
├── append_batch.py
├── data/
│   ├── holdout.csv.dvc
│   ├── train_batch1.csv.dvc
│   └── train_batch2.csv.dvc
├── deploy/income-api.service
├── docs/superpowers/
│   ├── plans/
│   └── specs/
├── params.yaml
├── prepare_data.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── serve.py
│   └── train.py
└── tests/
    ├── __init__.py
    ├── test_serve.py
    └── test_train.py
```

Generated CSV files, models, reports, MLflow state, virtual environments, DVC
local configuration, cloud credentials, and SSH private keys remain untracked.

## 4. Data Preparation and Versioning

`prepare_data.py` downloads the UCI Adult/Census Income train and test sources,
normalizes the two source label formats, removes rows containing `?`, selects
the ten required features, and alphabetically encodes the five categorical
features. It performs a deterministic shuffle with seed 42.

After cleaning, the script requires exactly 45,222 rows. It creates:

- `data/train_batch1.csv`: 22,361 rows.
- `data/holdout.csv`: 500 rows reserved only for evaluation.
- `data/train_batch2.csv`: 22,361 rows reserved for continuous training.

All files use these columns in order:

```text
age, workclass, education_num, marital_status, occupation, relationship,
sex, capital_gain, capital_loss, hours_per_week, target
```

`append_batch.py` appends batch 2 to batch 1 to produce 44,722 training rows. It
validates both schemas and refuses to append when batch 1 is already 44,722
rows, preventing accidental duplication.

DVC tracks the three CSV files. Git tracks only their `.dvc` pointers. The DVC
remote uses the selected bucket from Section 2 with the fixed `dvc/` prefix;
cloud credentials are not written to the tracked `.dvc/config`. Local-only DVC
credential configuration, when needed, uses ignored `.dvc/config.local`.

## 5. Training and Experiment Tracking

`src.train.train(params, data_path, eval_path) -> float`:

1. Validates the presence of `target`, identical train/evaluation features, and
   both target classes.
2. Trains `GradientBoostingClassifier(**params, random_state=42)`.
3. Evaluates only on the held-out dataset.
4. Computes positive-class `f1_score(y_eval, predictions)` without a macro or
   weighted average, plus reference accuracy.
5. Logs parameters, metrics, and the sklearn model to MLflow.
6. Writes `outputs/report.json` with numeric `f1_score` and `accuracy`.
7. Writes `models/model.joblib` and returns Python `float` F1.

MLflow uses local SQLite state at `sqlite:///mlflow.db` and local artifact state
under `mlartifacts/`. At least these three configurations are run:

| Run | n_estimators | learning_rate | max_depth |
|---|---:|---:|---:|
| Baseline | 100 | 0.10 | 3 |
| Weak/gate demonstration | 50 | 0.05 | 2 |
| Candidate | 200 | 0.10 | 5 |

The configuration with the strongest positive-class F1 that meets `F1 >= 0.65`
is retained in `params.yaml`. Actual results, rather than expected values, are
recorded in `REPORT.md`.

## 6. Inference Service

`src/serve.py` exposes a FastAPI application:

- `GET /healthz` returns `{"status": "ok"}` after the model is loaded.
- `POST /score` accepts `{"features": [10 numeric values]}`.
- A valid response contains integer `prediction` and label
  `thu_nhap_thap` for 0 or `thu_nhap_cao` for 1.
- A request with a feature count other than ten returns HTTP 400.

At service startup, the process downloads
`artifacts/current/model.joblib` from the configured bucket to a temporary local
path and atomically replaces the runtime model file. Download or model-load
failure terminates startup; systemd restarts the process after five seconds.
The model is downloaded once per process start, never per request.

The VM uses its attached GCP service identity for model download, so no service
account JSON key is copied to the VM. The tracked systemd unit supplies the
bucket and model-path environment variables and runs the API as the VM user.

## 7. CI/CD Pipeline

The workflow triggers on `main` changes to DVC pointers, Python source, tests,
dependencies, parameters, data scripts, or the workflow itself. Manual dispatch
is also enabled.

```text
Unit Test -> Train -> Quality Gate -> Release
```

### Unit Test

Checks out the repository, installs Python 3.10 dependencies, and runs the full
pytest suite using generated test data and mocked cloud interactions.

### Train

Authenticates to GCP using the `STORAGE_CREDENTIALS` GitHub secret, pulls only
batch 1 and holdout data with DVC, runs training, exposes F1 as a job output, and
uploads `model.joblib` plus `report.json` as a GitHub Actions artifact. It does
not overwrite the production model.

### Quality Gate

Parses the Train output explicitly as `float`. It fails when `f1_score < 0.65`
and therefore prevents Release. Accuracy is logged but never gates deployment,
because a majority-only classifier can achieve about 0.752 accuracy while its
positive-class F1 is zero.

### Release

Runs only after the gate succeeds. It downloads the exact model artifact from
Train, authenticates to GCP, uploads that file to
`artifacts/current/model.joblib`, connects to the VM over SSH, restarts
`income-api`, waits for readiness, and requires a successful local `/healthz`
response. This ordering prevents a rejected model from replacing the current
production artifact.

Required GitHub secrets are:

- `STORAGE_CREDENTIALS`
- `ARTIFACT_BUCKET`
- `SERVER_HOST`
- `SERVER_USER`
- `SERVER_SSH_KEY`

No secret value appears in logs, tracked files, reports, or documentation.

## 8. Tests and Failure Behavior

Training tests use deterministic, synthetic two-class data and isolated
temporary working directories. They verify:

- `train()` returns a float in `[0.0, 1.0]`.
- The JSON report has numeric `f1_score` and `accuracy`.
- The serialized model exists and can be loaded.
- Invalid schemas and one-class data fail before model publication.

Serving tests mock storage/model dependencies and verify health, both prediction
labels, and HTTP 400 for invalid feature length.

Data preparation fails on download errors, unexpected cleaned row counts,
unknown schema, or split-size mismatches. CI fails closed: a missing report,
unparseable F1, failed upload, failed SSH restart, or failed health check makes
the responsible job red.

## 9. Continuous-Training Demonstration

After the first successful deployment:

1. Run `append_batch.py` and verify the row count changes from 22,361 to 44,722.
2. Run `dvc add data/train_batch1.csv`.
3. Commit only `data/train_batch1.csv.dvc`.
4. Run `dvc push` before pushing Git.
5. Push the data-pointer commit to `main`.
6. Verify the same four jobs run and the service restarts with the promoted
   model when F1 remains at or above 0.65.

The before/after F1 values are recorded without assuming that doubling data from
the same distribution improves the metric.

## 10. Verification and Submission Evidence

Completion requires fresh evidence from:

- A local pytest run with all tests passing.
- Three MLflow runs with visible parameter, F1, and accuracy differences.
- One deliberately weak run where Quality Gate fails and Release is skipped,
  followed by restoration of the selected parameters.
- Two successful GitHub workflow runs: initial deployment and data-pointer
  continuous training.
- GCS object listings proving the `dvc/` prefix and current model.
- Live curl responses for `/healthz`, a valid low-income score request, a valid
  high-income score request, and an invalid-length request.
- `REPORT.md`, limited to approximately one A4 page, containing selected
  parameters, real metric comparisons, the F1-versus-accuracy explanation, and
  encountered difficulties with resolutions.

Console screenshots remain a user-visible submission artifact. Commands, run
URLs, object paths, and metric values are prepared in the repository so the
required screenshots can be captured without reconstructing evidence.

## 11. Out of Scope

- Authentication or rate limiting for the public demonstration API.
- Container orchestration, Kubernetes, or a Docker-based deployment.
- Online feature stores, model registries, drift detection, or scheduled
  retraining.
- Production-grade keyless GitHub Workload Identity Federation migration; the
  lab uses the required repository secret while keeping it out of tracked files.
- Automatic deletion of billable cloud resources. Teardown instructions are
  documented, but deletion requires explicit user confirmation.
