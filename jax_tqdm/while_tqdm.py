from time import sleep
from typing import Any, Callable, Optional, TypeVar, Sequence

from jax_tqdm.base import PBar, build_tqdm  # type: ignore

C = TypeVar("C")
Z = TypeVar("Z")

# Body function type for jax.lax.while_loop
WhileFn = Callable[[C], C]
# Same function but optionally wrapped with a progress-bar aware PBar object
WrappedWhileFn = WhileFn | Callable[[PBar[C]], PBar[C]]


def while_tqdm(  # noqa: D401 – function acts as a decorator factory
    n: Optional[int] = None,
    print_rate: Optional[int] = None,
    tqdm_type: str = "auto",
    *,
    print_s: Optional[Sequence[float]] = None,
    s_max: Optional[float] = None,
    last_is_postfix: bool = False,
    postfix_fmt_str: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> Callable[[WhileFn], WrappedWhileFn]:
    """Decorator factory that adds a tqdm progress-bar to a JAX ``while_loop``.

    Parameters
    ----------
    n : int | None
        Maximum number of iterations expected for the ``while_loop``. If using
        schedule-based mode via ``print_s``/``s_max``, ``n`` may be ``None``.
    print_rate : int | None, optional
        Update rate for the progress-bar.  If *None* (default), it will be set
        to *n // 20* so that roughly twenty updates are shown.
    tqdm_type : {"auto", "std", "notebook"}
        Which tqdm flavour to use.  See :func:`jax_tqdm.base.build_tqdm`.
    print_s : Sequence[float] | None
        Optional schedule of elapsed-time thresholds; enables schedule mode.
        Requires ``s_max``.
    s_max : float | None
        Maximum elapsed time for schedule-based mode; required if ``print_s``.
    last_is_postfix : bool
        If True and the carry is a tuple, treat its last element as postfix dict.
    **kwargs : Any
        Extra keyword arguments forwarded directly to :pyclass:`tqdm.tqdm`.

    Returns
    -------
    Callable
        A wrapper to be applied on the *body_fun* passed to
        :func:`jax.lax.while_loop`.

    Notes
    -----
    For the progress-bar to work, the *carry* of your ``while_loop`` **must**
    contain the current iteration counter as its first element, e.g.::

        init_carry = (0, <rest_of_state>)

        def cond(carry):
            i, _ = carry
            return i < n

        def body(carry):
            i, state = carry
            # ... do work ...
            return (i + 1, state)

    If you are already propagating a :class:`jax_tqdm.base.PBar` through nested
    loops, this decorator will seamlessly integrate with it.
    """

    # Detect schedule-based progress mode
    schedule_mode = print_s is not None
    update_progress_bar, close_tqdm = build_tqdm(
        n,
        print_rate,
        tqdm_type,
        print_s=print_s,
        s_max=s_max,
        postfix_fmt_str=postfix_fmt_str,
        **kwargs,
    )

    def _while_tqdm(func: WhileFn) -> WrappedWhileFn:  # noqa: D401
        """Internal decorator that injects the progress-bar logic."""

        def wrapper_progress_bar_body(carry: Z) -> Z:  # type: ignore[override]
            # Detect whether we are nested inside an existing PBar.
            if isinstance(carry, PBar):
                bar_id = carry.id
                inner_carry = carry.carry  # unwrap the actual loop carry
            else:
                bar_id = 0  # top-level bar
                inner_carry = carry

            # Extract iteration index or progress value depending on mode
            if schedule_mode:
                # First element of carry is elapsed time s
                if isinstance(inner_carry, tuple):
                    s = inner_carry[0]  # type: ignore[index]
                else:
                    s = inner_carry  # type: ignore[assignment]
                iter_num = 1  # dummy non-zero to avoid repeated init; base ensures define-if-needed
                progress = s  # pass elapsed time to base
            else:
                # Iteration-based: first element is the iteration counter
                if isinstance(inner_carry, tuple):
                    iter_num = inner_carry[0]
                else:
                    iter_num = inner_carry  # type: ignore[assignment]
                progress = None

            # Optional postfix is the last element of the inner_carry when enabled
            if last_is_postfix and isinstance(inner_carry, tuple):
                _maybe_postfix = inner_carry[-1]
                postfix = _maybe_postfix if isinstance(_maybe_postfix, dict) else {}
            else:
                postfix = {}

            # Update the progress-bar.  ``build_tqdm`` expects a tuple so we
            # wrap and unwrap accordingly.
            (inner_carry,) = update_progress_bar((inner_carry,), iter_num, bar_id, progress, postfix)

            # Call the original body function.
            result = func(inner_carry)  # type: ignore[arg-type]

            # Re-wrap the result if we were propagating a nested PBar.
            if isinstance(carry, PBar):
                result = PBar(id=bar_id, carry=result)  # type: ignore[assignment]

            # Potentially close the bar when *iter_num* reaches *n - 1*.
            return close_tqdm(result, iter_num, bar_id, progress)  # type: ignore[return-value]

        return wrapper_progress_bar_body  # type: ignore[return-value]

    return _while_tqdm


# -----------------------------------------------------------------------------
# Minimal usage example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from time import sleep 
    max_iters = 100000000

    @while_tqdm(max_iters)
    def body(carry: tuple[int, jnp.ndarray]) -> tuple[int, jnp.ndarray]:
        i, acc = carry
        return (i + 1, acc + i)

    def cond(carry):
        i, _ = carry
        return i < max_iters // 2  # the progress bar will stop prematurely, we only know an upper bound on i ahead of time. Also the bar is not removed then

    final_carry = jax.lax.while_loop(cond, body, (0, jnp.zeros(())))
    sleep(10)
    print("Final carry:", final_carry)
