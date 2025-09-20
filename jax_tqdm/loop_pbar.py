from typing import Any, Callable, Optional, TypeVar, Sequence

from .base import PBar, build_tqdm

C = TypeVar("C")
Z = TypeVar("Z")

LoopFn = Callable[[int, C], C]
WrappedLoopFn = LoopFn | Callable[[int, PBar[C]], PBar[C]]


def loop_tqdm(
    n: int,
    print_rate: Optional[int] = None,
    tqdm_type: str = "auto",
    *,
    print_s: Optional[Sequence[float]] = None,
    s_max: Optional[float] = None,
    postfix_fmt_str: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> Callable[[LoopFn], WrappedLoopFn]:
    """
    tqdm progress bar for a JAX fori_loop

    Parameters
    ----------
    n: int
        Number of iterations.
    print_rate: int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    print_s : Sequence[float] | None
        Optional elapsed-time thresholds; enables schedule mode and requires ``s_max``.
    s_max : float | None
        Maximum elapsed time for schedule-based mode; required if ``print_s``.
    **kwargs
        Extra keyword arguments to pass to tqdm.

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

    def _loop_tqdm(func: LoopFn) -> WrappedLoopFn:
        """
        Decorator that adds a tqdm progress bar to `body_fun`
        used in `jax.lax.fori_loop`.
        """

        def wrapper_progress_bar(i: int, val: Z) -> Z:
            if isinstance(val, PBar):
                bar_id = val.id
                val = val.carry
                i, val = update_progress_bar((i, val), i, bar_id, None, {})
                result = func(i, val)
                result = PBar(id=bar_id, carry=result)
                return close_tqdm(result, i, bar_id)
            else:
                i, val = update_progress_bar((i, val), i, 0, None, {})
                result = func(i, val)
                return close_tqdm(result, i, 0)

        return wrapper_progress_bar

    return _loop_tqdm
