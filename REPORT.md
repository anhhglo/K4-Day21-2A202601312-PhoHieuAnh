# Income Model MLOps Report

## Objective and delivery design

This lab trains an Adult Income classifier, records experiments in MLflow, and
automatically promotes a model only after a quality gate. The GitHub Actions workflow
first runs unit tests, trains from DVC-managed data, publishes the candidate model and
metrics as an artifact, requires positive-class F1 of at least `0.65`, then deploys the
approved model to Cloud Storage and the inference service. Production is the
`income-api` systemd service on VM `income-api` (project `track2-day16-01312`, zone
`us-central1-a`). Its external IP was removed and public firewall rule disabled after
the required endpoint evidence was collected. The production object is
`gs://k4-day21-2a202601312-phohieuanh/artifacts/current/model.joblib`.

## Experiments and model choice

| Configuration | n_estimators | learning_rate | max_depth | F1 score | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 100 | 0.1 | 3 | 0.7109004739336493 | 0.878 |
| Weak comparison | 50 | 0.05 | 2 | 0.6051282051282051 | 0.846 |
| Strong candidate | 200 | 0.1 | 5 | 0.7149321266968326 | 0.874 |

The selected configuration is the strong candidate (`200`, `0.1`, `5`). Its
F1 of `0.7149321266968326` is the best recorded experiment and clears the
release threshold. Accuracy is retained as a descriptive metric, but it is not the
promotion criterion: with an approximately 75/25 class split, a majority-only model
can achieve about `0.752` accuracy while having zero positive-class F1. F1 therefore
better protects the ability to identify the higher-income class, balancing precision
and recall rather than rewarding the majority outcome alone.

## CI/CD and endpoint evidence

The initial complete green workflow run is
[`32507086104`](https://github.com/anhhglo/K4-Day21-2A202601312-PhoHieuAnh/actions/runs/32507086104).
It completed successfully and produced the selected initial F1
`0.7149321266968326`. Deployment verification returned `{"status":"ok"}` from
`GET /healthz`. Functional scoring checks returned `0` / `thu_nhap_thap` for the
low-income sample and `1` / `thu_nhap_cao` for the high-income sample, demonstrating
both response mapping branches.

## Data expansion: 22,361 vs. 44,722 training records

Commit `abb5dcc` adds 22,361 records, expanding the training data from 22,361 to
44,722 records while retaining the holdout evaluation set. This change intentionally
triggers a new CI/CD training cycle rather than assuming that more data improves F1.
The corresponding workflow is
[`32508377146`](https://github.com/anhhglo/K4-Day21-2A202601312-PhoHieuAnh/actions/runs/32508377146).
All four jobs completed successfully. Its holdout F1 is `0.7354260089686099`
(accuracy `0.882`), an increase of about `0.0205` from the initial
`0.7149321266968326`. The improvement is modest because both batches came from the
same source distribution; the important result is that a data-pointer-only commit
completed the entire train, gate, and release path without a manual intermediate step.

## Quality-gate rejection evidence

Commit `2643466` deliberately set an unusable configuration (`1`, `0.01`, `1`).
Workflow run
[`32509743817`](https://github.com/anhhglo/K4-Day21-2A202601312-PhoHieuAnh/actions/runs/32509743817)
recorded F1 `0.0000` (local accuracy `0.752`), failed Quality Gate at the `0.65`
threshold, and skipped Release. The production object's generation remained
`1787333919166004` before and after the rejected run, proving that the weak candidate
was not promoted. Commit `f3e1ce7` then restored the selected `200`/`0.1`/`5`
configuration. The restoration workflow
[`32510047862`](https://github.com/anhhglo/K4-Day21-2A202601312-PhoHieuAnh/actions/runs/32510047862)
completed all four jobs successfully and promoted production object generation
`1787334701712627`.

## Difficulties and resolutions

The main issues were restricted network access during the UCI download, DVC's need
to manage its own data ignore entries, and ensuring a passing health check could not
hide a stale in-memory model. Network operations were rerun with scoped approval,
DVC-specific ignore exceptions preserved both tracking and CSV safety, and Release
now explicitly restarts `income-api` after promoting every approved model.
