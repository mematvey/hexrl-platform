import json
from pathlib import Path

import pytest

from scripts.observability.analysis import (
    error_budget,
    first_time,
    good_fraction,
    histogram_series,
    label_cardinality,
)
from scripts.observability.stand import (
    Report,
    Stand,
    mebibytes,
    parse_compose_ps,
    readiness,
    task_path,
    trace_summary,
)


def container(service, state="running", health="healthy", exit_code=0):
    return {"Service": service, "State": state, "Health": health, "ExitCode": exit_code}


def stand_containers(workers=1, bootstrap_exit=0):
    items = [
        container(name)
        for name in ("db", "rabbitmq", "mlflow", "app", "grafana", "prometheus", "loki", "tempo")
    ]
    items += [container("worker") for _ in range(workers)]
    items += [container("alloy", health=""), container("otel-collector", health="")]
    items += [
        container("artifact-bucket", "exited", ""),
        container("model-bootstrap", "exited", "", bootstrap_exit),
    ]
    return items


def test_compose_ps_output_of_both_formats_is_parsed():
    rows = [container("app"), container("worker")]
    assert parse_compose_ps(json.dumps(rows)) == rows
    assert parse_compose_ps("\n".join(json.dumps(row) for row in rows)) == rows
    assert parse_compose_ps("  ") == []


def test_stand_is_ready_only_after_bootstrap_and_healthchecks():
    assert readiness(stand_containers()) == (True, [], [])
    ready, waiting, failed = readiness(stand_containers(), workers=2)
    assert not ready and waiting == ["worker is not healthy"] and not failed
    ready, _, failed = readiness(stand_containers(bootstrap_exit=1))
    assert not ready and failed == ["model-bootstrap exited with an error"]
    starting = [item for item in stand_containers() if item["Service"] != "model-bootstrap"]
    starting.append(container("model-bootstrap"))
    assert "model-bootstrap has not finished" in readiness(starting)[1]


def test_clean_up_is_refused_outside_the_isolated_stand(tmp_path):
    stand = Stand(tmp_path / ".env", project="hexrl-platform")
    with pytest.raises(ValueError, match="refusing"):
        import asyncio

        asyncio.run(stand.remove_volumes())


def test_up_starts_batch_source_only_when_named():
    stand = Stand(Path(".env.obs"))
    assert "batch" not in stand.command("up", "-d")
    assert "batch" in stand.command("up", "-d", "batch-source")
    assert "batch" in stand.command("down", "--volumes")


@pytest.mark.parametrize("key", ["batches", "trace"])
def test_trace_summary_reads_both_tempo_formats(key):
    resource = {
        "resource": {
            "attributes": [
                {"key": "service.name", "value": {"stringValue": "hexrl-worker"}},
                {"key": "service.instance.id", "value": {"stringValue": "abc"}},
            ]
        },
        "scopeSpans": [
            {
                "spans": [
                    {"name": "inference", "status": {}},
                    {"name": "process", "status": {"code": "STATUS_CODE_ERROR"}},
                ]
            }
        ],
    }
    payload = (
        {"batches": [resource]} if key == "batches" else {"trace": {"resourceSpans": [resource]}}
    )
    assert trace_summary(payload) == {
        "services": ["hexrl-worker"],
        "instances": ["abc"],
        "spans": ["inference", "process"],
        "error_spans": ["process"],
    }


def test_docker_memory_units():
    assert mebibytes("512MiB") == 512
    assert mebibytes("1.5GiB") == 1536
    assert mebibytes("2048KiB") == 2
    with pytest.raises(ValueError):
        mebibytes("12 parsecs")


def test_queue_is_full_from_the_first_sample_at_the_threshold():
    assert first_time([(0, 3), (5, 18), (10, 20)], 18, after=1) == 5
    assert first_time([(0, 20), (5, 3)], 18, after=1) is None
    assert first_time([(0, 3)], 18) is None


def test_slo_budget_and_cardinality_arithmetic():
    assert good_fraction(990, 1000) == 0.99
    assert good_fraction(0, 0) is None
    assert error_budget(6000, 0.99) == pytest.approx(60)
    with pytest.raises(ValueError):
        error_budget(10, 1)
    # 13 границ task.duration: 13 + Inf + sum + count на одно сочетание labels
    assert histogram_series(13, 1) == 16
    assert histogram_series(13, label_cardinality(6000)) == 96_000
    assert label_cardinality(2, 3, 4) == 24


def test_report_fails_when_any_check_fails(tmp_path):
    report = Report("demo")
    assert not report.ok
    report.check("first", True)
    assert report.ok
    report.check("second", False, detail=1)
    saved = json.loads(report.save(tmp_path).read_text(encoding="utf-8"))
    assert saved["ok"] is False and saved["checks"][1] == {
        "check": "second",
        "ok": False,
        "detail": 1,
    }


def test_task_path_skips_status_polls_and_untraced_lines():
    records = [
        {"event": "http_request_completed", "method": "POST", "trace_id": "a"},
        {"event": "http_request_completed", "method": "GET", "trace_id": "b"},
        {"event": '127.0.0.1 - "GET /api/v1/tasks/x HTTP/1.1" 200'},
        {"event": "task_completed", "trace_id": "a"},
    ]
    assert [r["trace_id"] for r in task_path(records)] == ["a", "a"]
