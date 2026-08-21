# Day 21 submission evidence

Use the files in this exact order:

1. [`01-mlflow-experiments.png`](01-mlflow-experiments.png) — direct Chrome
   screenshot of the local MLflow `adult-income` experiment. Four real runs are
   visible together with `accuracy`, `f1_score`, `learning_rate`, `max_depth`, and
   `n_estimators`.
2. [`02-actions-initial-green.png`](02-actions-initial-green.png) — direct screenshot
   of GitHub Actions run `32507086104`; all four jobs are green.
3. [`03-actions-data-commit-green.png`](03-actions-data-commit-green.png) — direct
   screenshot of data-pointer commit run `32508377146`; all four jobs are green.
4. [`04-endpoint-curl-iap.png`](04-endpoint-curl-iap.png) — terminal rendering of
   commands executed on the live VM through GCP IAP. The public IP remains disabled;
   the commands use private VM IP `10.128.0.2` and show health plus both score labels.
5. [`05-cloud-storage-dvc-model.png`](05-cloud-storage-dvc-model.png) — terminal
   rendering of authenticated `gcloud storage` output. It shows the real `dvc/`
   objects, their mapping to the three `.dvc` pointers, and
   `artifacts/current/model.joblib` metadata.
6. [`06-actions-quality-gate-failed.png`](06-actions-quality-gate-failed.png) — direct
   screenshot of run `32509743817`: Unit Test and Train passed, Quality Gate failed,
   and Release was skipped.
7. [`07-bao-cao-1-trang.pdf`](07-bao-cao-1-trang.pdf) — Vietnamese one-page A4
   report. The PNG preview is
   [`07-bao-cao-1-trang.png`](07-bao-cao-1-trang.png), and the editable source is
   [`../bao-cao-1-trang.html`](../bao-cao-1-trang.html).

## Provenance and limitation

No AI-generated or fabricated UI is used. Items 1, 2, 3, and 6 are direct browser
screenshots. Items 4 and 5 are clearly labelled terminal renderings of outputs that
were freshly executed against the live VM and authenticated bucket.

The automated Chrome profile was not signed in to Google Cloud Console. If the
submission rubric strictly requires the Console UI rather than authenticated CLI
evidence, open the bucket in your signed-in browser and replace item 5 with a manual
screenshot. Do not re-enable or publish an external VM IP merely for screenshots.
