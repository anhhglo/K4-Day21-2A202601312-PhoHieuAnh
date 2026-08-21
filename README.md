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
project `track2-day16-01312`, zone `us-central1-a`. Public access was disabled after
the lab verification: the VM has no external IP and the former public `8080` firewall
rule is disabled. The GitHub deployment host and SSH-key secrets were also removed,
so future Release jobs remain locked until access is deliberately reconfigured.
Promoted model artifacts are stored in
`gs://k4-day21-2a202601312-phohieuanh/artifacts/current/model.joblib`.

Secrets are supplied through GitHub Actions and the VM environment file. Do not put
service-account keys, SSH private keys, or GitHub secret values in the repository or
in shell history.

## Deploy and verify

The normal deployment path is a push to `main` that changes a pipeline input (data
pointer, training code, tests, parameters, requirements, or workflow). GitHub Actions
trains a candidate, blocks release unless positive-class F1 is at least `0.65`, copies
the approved model to Cloud Storage, and restarts the systemd service.

These non-secret commands are useful for confirming the retained artifacts. Direct
public `curl` access is intentionally unavailable while the VM is locked.

```bash
# Check the promoted artifact exists.
gcloud storage ls gs://k4-day21-2a202601312-phohieuanh/artifacts/current/model.joblib

# Confirm that no external NAT address remains attached.
gcloud compute instances describe income-api --project=track2-day16-01312 \
  --zone=us-central1-a --format='yaml(networkInterfaces[0].accessConfigs)'
```

The successful Actions runs retain the original health and scoring evidence. Reopen
network access only temporarily and only for an explicitly approved source CIDR.

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
