"""Test closed-form advance: must match brute-force step summation."""

from fractions import Fraction

from dryrun.component.rollout.engine.advance import crossing_step, elapsed, next_completion, steps_within


def test_elapsed_matches_brute_force():
    F, alpha, beta = Fraction(5), Fraction(2), Fraction(1)
    for k in range(20):
        brute = sum(max(F, alpha + beta * i) for i in range(k))
        assert elapsed(F, alpha, beta, k) == brute, f"Mismatch at k={k}."


def test_elapsed_no_floor():
    F, alpha, beta = Fraction(0), Fraction(3), Fraction(2)
    for k in range(15):
        brute = sum(alpha + beta * i for i in range(k))
        assert elapsed(F, alpha, beta, k) == brute


def test_elapsed_pure_floor():
    F, alpha, beta = Fraction(10), Fraction(1), Fraction(0)
    for k in range(10):
        assert elapsed(F, alpha, beta, k) == F * k


def test_steps_within_inverse_of_elapsed():
    F, alpha, beta = Fraction(5), Fraction(2), Fraction(1)
    for k in range(1, 20):
        budget = elapsed(F, alpha, beta, k)
        assert steps_within(F, alpha, beta, budget) == k
        assert steps_within(F, alpha, beta, budget - Fraction(1, 1000)) == k - 1


def test_steps_within_float():
    F, alpha, beta = 5.0, 2.0, 1.0
    for k in range(1, 15):
        budget = float(elapsed(Fraction(5), Fraction(2), Fraction(1), k))
        result = steps_within(F, alpha, beta, budget)
        assert result == k or result == k - 1


def test_next_completion():
    assert next_completion([]) == (0, 0)
    assert next_completion([5, 3, 3, 7]) == (3, 2)
    assert next_completion([1]) == (1, 1)
    assert next_completion([4, 4, 4]) == (4, 3)


def test_crossing_step_basics():
    assert crossing_step(Fraction(10), Fraction(10), Fraction(1)) == 0
    assert crossing_step(Fraction(10), Fraction(5), Fraction(1)) == 5
    assert crossing_step(Fraction(10), Fraction(5), Fraction(2)) == 3
    assert crossing_step(Fraction(10), Fraction(5), Fraction(0)) == 0
