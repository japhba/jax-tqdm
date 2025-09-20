from typing import Any, Callable, Optional, TypeVar, Sequence

from .base import PBar, build_tqdm

C = TypeVar("C")
X = TypeVar("X", int, tuple[int, Any])
Y = TypeVar("Y")
Z = TypeVar("Z")

ScanFn = Callable[[C, X], tuple[C, Y]]
WrappedScanFn = ScanFn | Callable[[PBar[C], X], tuple[PBar[C], Y]]


def scan_tqdm(
    n: Optional[int] = None,
    print_rate: Optional[int] = None,
    tqdm_type: str = "auto",
    *,
    print_s: Optional[Sequence[float]] = None,
    s_max: Optional[float] = None,
    last_is_postfix: bool = False,
    postfix_fmt_str: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> Callable[[ScanFn], WrappedScanFn]:
    """
    tqdm progress bar for a JAX scan

    Parameters
    ----------
    n : int | None
        Number of scan steps/iterations. If using schedule-based mode via
        ``print_s``/``s_max``, ``n`` may be ``None``.
    print_rate : int | None
        Optional integer rate at which the progress bar will be updated; by
        default the print rate will be 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    print_s : Sequence[float] | None
        Optional elapsed-time thresholds; enables schedule mode and requires ``s_max``.
    s_max : float | None
        Maximum elapsed time for schedule-based mode; required if ``print_s``.
    last_is_postfix : bool
        If True, interpret last element of carry tuple as postfix dict.
    **kwargs
        Extra keyword arguments forwarded to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    update_progress_bar, close_tqdm = build_tqdm(
        n,
        print_rate,
        tqdm_type,
        print_s=print_s,
        s_max=s_max,
        postfix_fmt_str=postfix_fmt_str,
        **kwargs,
    )

    def _scan_tqdm(func: ScanFn) -> WrappedScanFn:
        """Decorator that adds a tqdm progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `jnp.arange(n)`,
        or be looping over a tuple who's first element is `jnp.arange(n)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry: Z, x: X) -> tuple[Z, Y]:
            if isinstance(x, tuple):
                iter_num, *_ = x
            else:
                iter_num = x
            
            # Safely extract postfix when enabled and present
            try:
                postfix = carry[-1]  # type: ignore[index]
            except Exception:
                postfix = {}

            if isinstance(carry, PBar):
                bar_id = carry.id
                carry_ = carry.carry
                carry_, x = update_progress_bar((carry_, x), iter_num, bar_id, None, postfix if isinstance(postfix, dict) and last_is_postfix else {})
                result = func(carry_, x)
                result = (PBar(id=bar_id, carry=result[0]), result[1])
                return close_tqdm(result, iter_num, bar_id)
            else:
                carry, x = update_progress_bar((carry, x), iter_num, 0, None, postfix if isinstance(postfix, dict) and last_is_postfix else {})
                result = func(carry, x)
                return close_tqdm(result, iter_num, 0)

        return wrapper_progress_bar

    return _scan_tqdm
