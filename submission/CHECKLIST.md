# Submission Screenshot Checklist

Capture screenshots in this order. Include the browser or terminal context that makes
the project, resource, and result identifiable; redact any credentials or tokens.

1. **Repository overview** — repository name, default branch, and the README pipeline
   description.
2. **Versioned data** — DVC pointer files for `train_batch1.csv`, `train_batch2.csv`,
   and `holdout.csv`, plus the commit `abb5dcc` showing the data-only update.
3. **Experiment tracking** — MLflow experiment view showing the baseline, weak, and
   strong candidate runs and their F1 metrics.
4. **Initial CI/CD success** — GitHub Actions run
   `32507086104`, with all four jobs (`unit-test`, `train`, `quality-gate`, `release`)
   green and the F1 gate evidence visible.
5. **Data-expansion CI/CD run** — GitHub Actions run `32508377146` for commit
   `abb5dcc`; capture all four green jobs and the new F1 `0.7354260089686099`.
6. **Quality Gate rejection** — GitHub Actions run `32509743817` for commit
   `2643466`; capture Unit Test and Train passing, Quality Gate failing with
   `f1_score 0.0000 < 0.65`, and Release skipped.
7. **Cloud artifact** — Cloud Storage object
   `artifacts/current/model.joblib` in bucket `k4-day21-2a202601312-phohieuanh`.
8. **Compute deployment** — Compute Engine VM `income-api` in project
   `track2-day16-01312`, zone `us-central1-a`. Public access was removed after the
   endpoint evidence was collected; do not publish a replacement IP.
9. **Service health** — terminal or API client response from `GET /healthz` showing
   `{"status":"ok"}` and, if available, `systemctl status income-api`.
10. **Low-income score** — `POST /score` response containing `prediction: 0` and
   `label: thu_nhap_thap`.
11. **High-income score** — `POST /score` response containing `prediction: 1` and
    `label: thu_nhap_cao`.
12. **Final evidence set** — a final view that connects the successful workflow,
    restored-parameter run `32510047862`, promoted Cloud Storage artifact,
    VM/service, and both score responses. Verify no secret values are visible before
    upload.
