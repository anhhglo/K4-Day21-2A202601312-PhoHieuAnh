# Income Model MLOps CI/CD Lab

This lab automates the Adult Income model lifecycle: deterministic data preparation,
training and experiment tracking, quality-gated model promotion, and API deployment.

## Requirements

Use Python 3.10 or newer. Deployment requires a positive-class F1 score of at least
0.65; accuracy is not a deployment gate.

## Pipeline

The GitHub Actions pipeline has four jobs: `unit-test`, `train`, `quality-gate`, and
`release`.

## Deployed environment

The released service runs as `income-api` on Compute Engine VM `income-api` in
project `track2-day16-01312`, zone `us-central1-a`. The VM public IP is
`34.42.113.234`; promoted model artifacts are stored in
`gs://k4-day21-2a202601312-phohieuanh/artifacts/current/model.joblib`.

Secrets are supplied through GitHub Actions and the VM environment file. Do not put
service-account keys, SSH private keys, or GitHub secret values in the repository or
in shell history.

## Deploy and verify

The normal deployment path is a push to `main` that changes a pipeline input (data
pointer, training code, tests, parameters, requirements, or workflow). GitHub Actions
trains a candidate, blocks release unless positive-class F1 is at least `0.65`, copies
the approved model to Cloud Storage, and restarts the systemd service.

These non-secret commands are useful for confirming the deployment:

```bash
# Inspect the deployed service (requires an authenticated gcloud account).
gcloud compute ssh income-api --project=track2-day16-01312 --zone=us-central1-a \
  --command='sudo systemctl status income-api --no-pager; curl --fail http://127.0.0.1:8080/healthz'

# Check the promoted artifact exists.
gcloud storage ls gs://k4-day21-2a202601312-phohieuanh/artifacts/current/model.joblib

# Verify the public health endpoint.
curl --fail http://34.42.113.234:8080/healthz

# Score a ten-feature observation; the response contains prediction and label.
curl --fail -X POST http://34.42.113.234:8080/score \
  -H 'content-type: application/json' \
  -d '{"features":[39,0,13,0,0,40,0,0,0,0]}'
```

## Teardown (cost control)

Run these commands only when the lab is no longer needed. They delete the VM and
stored artifacts, so archive required evidence first.

```bash
gcloud compute instances delete income-api --project=track2-day16-01312 \
  --zone=us-central1-a --quiet
gcloud storage rm --recursive \
  gs://k4-day21-2a202601312-phohieuanh/artifacts/
# Optional: remove the entire lab bucket after confirming nothing else uses it.
gcloud storage rm --recursive gs://k4-day21-2a202601312-phohieuanh
```
