import pytest

from server.backend.base import StepOutput
from server.backend.fake import FakeBackend


def make_backend(tokens_per_sequence=3):
    return FakeBackend(tokens_per_sequence=tokens_per_sequence, step_seconds=0)


def test_step_returns_one_token_for_each_active_sequence():
    backend = make_backend()
    first = backend.start_sequence([1, 2])
    second = backend.start_sequence([10])

    assert backend.step().keys() == {first, second}


def test_tokens_are_deterministic_across_backends():
    def generate(backend):
        seq_id = backend.start_sequence([1, 2])
        return [backend.step()[seq_id].token for _ in range(3)]

    assert generate(make_backend()) == generate(make_backend())


def test_tokens_do_not_depend_on_other_active_sequences():
    alone = make_backend()
    alone_id = alone.start_sequence([1, 2])

    shared = make_backend()
    shared_id = shared.start_sequence([1, 2])
    shared.start_sequence([50, 60])

    for _ in range(3):
        assert alone.step()[alone_id] == shared.step()[shared_id]


def test_sequence_finishes_on_its_nth_token():
    backend = make_backend(tokens_per_sequence=3)
    seq_id = backend.start_sequence([1])

    finished = [backend.step()[seq_id].finished for _ in range(3)]

    assert finished == [False, False, True]


def test_finished_sequence_is_not_stepped_again():
    backend = make_backend(tokens_per_sequence=1)
    seq_id = backend.start_sequence([1])
    assert backend.step()[seq_id] == StepOutput(token=1, finished=True)

    assert backend.step() == {}


def test_sequences_started_later_finish_later():
    backend = make_backend(tokens_per_sequence=2)
    early = backend.start_sequence([1])
    backend.step()
    late = backend.start_sequence([2])

    outputs = backend.step()

    assert outputs[early].finished is True
    assert outputs[late].finished is False
    assert backend.step()[late].finished is True


def test_released_sequence_is_not_stepped():
    backend = make_backend()
    released = backend.start_sequence([1])
    kept = backend.start_sequence([2])

    backend.release_sequence(released)

    assert backend.step().keys() == {kept}


def test_releasing_unknown_sequence_raises():
    with pytest.raises(KeyError):
        make_backend().release_sequence(999)


def test_step_sleeps_once_regardless_of_sequence_count(monkeypatch):
    sleeps = []
    monkeypatch.setattr("server.backend.fake.time.sleep", sleeps.append)
    backend = FakeBackend(tokens_per_sequence=3, step_seconds=0.25)
    backend.start_sequence([1])
    backend.start_sequence([2])

    backend.step()

    assert sleeps == [0.25]


def test_tokenize_is_deterministic_and_nonempty():
    backend = make_backend()

    assert backend.tokenize("ab") == backend.tokenize("ab")
    assert backend.tokenize("ab") != backend.tokenize("ba")
    assert backend.tokenize("ab")


def test_detokenize_returns_bytes_for_each_token():
    backend = make_backend()

    assert backend.detokenize([1]) != backend.detokenize([2])
    assert isinstance(backend.detokenize([1]), bytes)
