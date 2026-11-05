from server.kv_budget import BlockBudget, KVBudget


def test_reservation_that_fits_reduces_available():
    budget = KVBudget(100)

    assert budget.try_reserve(60) is True
    assert budget.available == 40


def test_reservation_of_exactly_what_is_left_succeeds():
    budget = KVBudget(100)

    assert budget.try_reserve(100) is True
    assert budget.available == 0


def test_refused_reservation_leaves_available_unchanged():
    budget = KVBudget(100)
    budget.try_reserve(60)

    assert budget.try_reserve(50) is False
    assert budget.available == 40


def test_release_makes_tokens_available_again():
    budget = KVBudget(100)
    budget.try_reserve(60)

    budget.release(60)

    assert budget.available == 100
    assert budget.try_reserve(100) is True


def test_fits_ever_compares_against_total_capacity():
    budget = KVBudget(100)
    budget.try_reserve(90)

    assert budget.fits_ever(100) is True
    assert budget.fits_ever(101) is False


def test_block_budget_rounds_tokens_up_to_whole_blocks():
    budget = BlockBudget(num_blocks=4, block_size=16)

    assert budget.try_reserve(17) is True
    assert budget.available == 2


def test_block_budget_refuses_when_blocks_run_out():
    budget = BlockBudget(num_blocks=2, block_size=16)
    budget.try_reserve(16)

    assert budget.try_reserve(17) is False
    assert budget.available == 1


def test_block_budget_release_returns_the_same_blocks():
    budget = BlockBudget(num_blocks=4, block_size=16)
    budget.try_reserve(17)

    budget.release(17)

    assert budget.available == 4


def test_block_budget_fits_ever_counts_total_blocks():
    budget = BlockBudget(num_blocks=2, block_size=16)

    assert budget.fits_ever(32) is True
    assert budget.fits_ever(33) is False
