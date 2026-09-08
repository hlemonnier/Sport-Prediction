"""Keep the portable test suite tied to the immutable experiment-time tests."""
import hashlib
from pathlib import Path


def test_only_batch_shape_roundoff_assertion_differs_from_frozen_suite():
    here = Path(__file__).resolve().parent
    frozen = (here.parent / "execution/test_model.py").read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == (
        "89fa364c5f9157a82f1df27b072d23e45ef6e4c301bb179dce8cf2f5dce9f6fe"
    )
    exact = b"np.testing.assert_array_equal(model.predict(rows[:1] + [distractor])[:1], reference)"
    portable = (
        b"np.testing.assert_allclose(model.predict(rows[:1] + [distractor])[:1], "
        b"reference, rtol=0, atol=8 * np.finfo(np.float64).eps)"
    )
    assert frozen.count(exact) == 1
    assert (here / "test_model.py").read_bytes() == frozen.replace(exact, portable)
