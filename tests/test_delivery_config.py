from pathlib import Path

import yaml


WORKFLOW_PATH = Path(".github/workflows/cicd.yml")
SERVICE_PATH = Path("deploy/income-api.service")


def test_workflow_promotes_only_a_quality_gated_candidate():
    workflow = yaml.load(WORKFLOW_PATH.read_text(), Loader=yaml.BaseLoader)
    jobs = workflow["jobs"]

    assert list(jobs) == ["unit-test", "train", "quality-gate", "release"]
    assert jobs["train"]["needs"] == "unit-test"
    assert jobs["quality-gate"]["needs"] == "train"
    assert jobs["release"]["needs"] == "quality-gate"
    assert "float(" in jobs["quality-gate"]["steps"][0]["run"]
    assert "artifacts/current/model.joblib" not in str(jobs["train"])
    assert "artifacts/current/model.joblib" in str(jobs["release"])


def test_release_preloads_the_vm_host_key_before_strict_ssh_deployment():
    workflow = WORKFLOW_PATH.read_text()

    assert "ssh-keyscan -H \"$SERVER_HOST\"" in workflow
    assert "StrictHostKeyChecking=yes" in workflow


def test_release_restarts_income_api_after_installing_the_new_service_unit():
    workflow = WORKFLOW_PATH.read_text()

    assert "sudo systemctl enable income-api" in workflow
    assert "sudo systemctl restart income-api" in workflow
    assert "sudo systemctl enable --now income-api" not in workflow


def test_income_api_service_restarts_with_documented_runtime_configuration():
    service = SERVICE_PATH.read_text()

    assert "Restart=always" in service
    assert "RestartSec=5" in service
    assert "--port 8080" in service
    assert "EnvironmentFile=/etc/income-api.env" in service
