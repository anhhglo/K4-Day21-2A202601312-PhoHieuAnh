# Income Model MLOps CI/CD Lab

This lab automates the Adult Income model lifecycle: deterministic data preparation,
training and experiment tracking, quality-gated model promotion, and API deployment.

## Requirements

Use Python 3.10 or newer. Deployment requires a positive-class F1 score of at least
0.65; accuracy is not a deployment gate.

## Pipeline

The GitHub Actions pipeline has four jobs: `unit-test`, `train`, `quality-gate`, and
`release`.
