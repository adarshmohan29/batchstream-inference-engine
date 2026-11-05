from server.metrics import RequestMetrics


def test_ttft_is_first_token_minus_arrival():
    metrics = RequestMetrics(arrival=10.0)

    metrics.record_token(12.5)

    assert metrics.ttft == 2.5


def test_later_tokens_do_not_change_first_token_time():
    metrics = RequestMetrics(arrival=10.0)

    metrics.record_token(12.5)
    metrics.record_token(13.0)

    assert metrics.first_token == 12.5


def test_end_to_end_is_finish_minus_arrival():
    metrics = RequestMetrics(arrival=10.0)

    metrics.record_finish(14.0)

    assert metrics.end_to_end == 4.0
