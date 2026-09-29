from lexeu.core.logging import add_severity


def test_json_logs_carry_a_cloud_logging_severity() -> None:
    event = add_severity(None, "warning", {"event": "x", "level": "warning"})
    assert event["severity"] == "WARNING"
