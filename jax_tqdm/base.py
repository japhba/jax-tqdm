from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar, Sequence

import jax
import tqdm.auto
import tqdm.notebook
import tqdm.std
from jax.debug import callback
import numpy as np
import numbers

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

C = TypeVar("C")
A = TypeVar("A")
B = TypeVar("B")

UpdateProgressBar = Callable[[A, int, int, Optional[float], dict], A]
CloseTQDM = Callable[[B, int, int, Optional[float]], B]


@dataclass
class PBar(Generic[C]):
    id: int
    carry: C


def build_tqdm(
    n: Optional[int] = None,
    print_rate: Optional[int] = None,
    tqdm_type: str = "auto",
    *,
    print_s: Optional[Sequence[float]] = None,
    s_max: Optional[float] = None,
    postfix_fmt_str: Optional[dict[str, str]] = None,
    **kwargs: Any,
) -> tuple[UpdateProgressBar, CloseTQDM]:
    """
    Build the tqdm progress bar on the host

    Parameters
    ----------
    n: int | None
        Number of updates for iteration-based mode. If using schedule-based
        mode via ``print_s``/``s_max``, ``n`` may be ``None``.
    print_rate: int | None
        Optional integer rate at which the progress bar will be updated.
        If ``None`` (default) the print rate will be 1/20th of the total
        number of steps.
    tqdm_type: str
        Type of progress-bar, should be one of "auto", "std", or "notebook".
    print_s: Sequence[float] | None
        Optional schedule of elapsed-time thresholds at which to update the bar.
        Enables schedule-based mode; requires ``s_max``.
    s_max: float | None
        Maximum elapsed time for the schedule-based bar. Required if
        ``print_s`` is provided.
    postfix_fmt_str: dict[str, str] | None
        Optional per-key Python format specs (e.g., {"loss": ".3e", "acc": ".2%"})
        that override the default postfix formatting.
    **kwargs
        Extra keyword arguments forwarded to :pyclass:`tqdm.tqdm`.
    """

    if tqdm_type not in ("auto", "std", "notebook"):
        raise ValueError(
            'tqdm_type should be one of "auto", "std", or "notebook" '
            f'but got "{tqdm_type}"'
        )
    pbar = getattr(tqdm, tqdm_type).tqdm

    # Optional schedule-based updates: print when elapsed time s crosses print_s; bar ends at s_max
    schedule_mode = print_s is not None

    if schedule_mode and s_max is None:
        raise ValueError("When using print_s, you must also provide s_max.")

    if schedule_mode:
        print_s_arr = np.sort(np.asarray(print_s, dtype=float).ravel())
        total = float(s_max)  # tqdm total measured in units of s
    else:
        if n is None:
            raise ValueError(
                "n must be provided when not using schedule mode (print_s/s_max)."
            )
        print_s_arr = None  # type: ignore[assignment]
        total = int(n)

    # Postfix formatting controls
    format_postfix_kw = kwargs.pop("format_postfix", None)
    legacy_format_kw = kwargs.pop("format", None)
    # Default to formatting postfix values unless explicitly disabled
    format_postfix = format_postfix_kw if format_postfix_kw is not None else (legacy_format_kw if legacy_format_kw is not None else True)
    postfix_decimals = int(kwargs.pop("postfix_decimals", 4))

    desc = kwargs.pop("desc", (f"Running to s_max={total:g}" if schedule_mode else f"Running for {n:,} iterations"))
    message = kwargs.pop("message", desc)
    position_offset = kwargs.pop("position", 0)
    postfix = kwargs.pop("postfix", {})

    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = dict()

    if not schedule_mode:
        if print_rate is None:
            if n > 20:
                print_rate = int(n / 20)
            else:
                print_rate = 1
        else:
            if print_rate < 1:
                raise ValueError(f"Print rate should be > 0 got {print_rate}")
            elif print_rate > n:
                raise ValueError(
                    "Print rate should be less than the "
                    f"number of steps {n}, got {print_rate}"
                )
        remainder = n % print_rate
        remainder = remainder if remainder > 0 else print_rate
    else:
        # Not used in schedule mode; closing will advance to total explicitly
        print_rate = 1
        remainder = 0

    def _define_tqdm(bar_id: int) -> None:
        bar_id = int(bar_id)
        if bar_id not in tqdm_bars:
            tqdm_bars[bar_id] = pbar(
                total=total,
                position=bar_id + position_offset,
                desc=message,
                **kwargs,
            )

    def _update_tqdm(bar_id: int, postfix_dict, s: Optional[float] = None) -> None:
        _bar = tqdm_bars[int(bar_id)]
        if schedule_mode:
            # Progress is measured in elapsed time s; update only when crossing scheduled thresholds
            if s is None:
                return
            # Clamp and round to stabilize number of displayed decimals
            s_cur = float(min(max(s, 0.0), total))
            s_cur_round = round(s_cur, postfix_decimals)
            prev = float(_bar.n)
            # Use rounded values consistently for thresholding and updating
            prev_round = round(prev, postfix_decimals)
            delta = s_cur_round - prev_round
            # Determine if any thresholds in (prev, s_cur] were crossed
            idx_prev = int(np.searchsorted(print_s_arr, prev_round, side='right'))
            idx_cur = int(np.searchsorted(print_s_arr, s_cur_round, side='right'))
            should_update = (idx_cur > idx_prev) and (delta > 0)
            if not should_update:
                return
            _bar.update(delta)
            # Keep displayed 's' aligned with the bar after update
            try:
                postfix_dict = dict(postfix_dict)
                postfix_dict['s'] = s_cur_round
            except Exception:
                pass
        else:
            _bar.update(print_rate)

        fmt_postfix = {}
        if format_postfix:
            for k, v in postfix_dict.items():
                try:
                    # Convert 0-dim arrays/scalars
                    v_scalar = np.asarray(v).item() if hasattr(v, "shape") and np.asarray(v).size == 1 else v

                    # Apply per-key override if provided
                    if postfix_fmt_str and k in postfix_fmt_str and postfix_fmt_str[k]:
                        fmt = postfix_fmt_str[k]
                        try:
                            fmt_postfix[k] = format(v_scalar, fmt)
                            continue
                        except Exception:
                            # Fall through to defaults on formatting error
                            pass

                    # Default formatting
                    if isinstance(v_scalar, numbers.Integral):
                        fmt_postfix[k] = f"{v_scalar}"
                    else:
                        try:
                            v_float = float(v_scalar)
                            fmt_postfix[k] = f"{v_float:.{postfix_decimals}f}"
                        except Exception:
                            fmt_postfix[k] = f"{v_scalar}"
                except Exception:
                    fmt_postfix[k] = f"{v}"
        else:
            fmt_postfix = postfix_dict
        tqdm_bars[0].set_postfix(fmt_postfix)

    def _close_tqdm(bar_id: int) -> None:
        _pbar = tqdm_bars.pop(int(bar_id))
        # In schedule mode, complete the bar to total; otherwise, add remainder
        if schedule_mode:
            remaining = float(total) - float(_pbar.n)
            if remaining > 0:
                _pbar.update(remaining)
        else:
            _pbar.update(remainder)
        _pbar.clear()
        _pbar.close()

    def update_progress_bar(carry: A, iter_num: int, bar_id: int, progress: Optional[float] = None, postfix: dict = {}) -> A:
        """Updates tqdm from a JAX scan or loop"""

        def _inner_init(_i: int, _carry: A) -> A:
            callback(_define_tqdm, bar_id, ordered=True)
            return _carry

        def _inner_update(i: int, _carry: A) -> A:
            if schedule_mode:
                # Ensure bar exists, then update using elapsed-time progress
                _ = callback(_define_tqdm, bar_id, ordered=True)
                _ = callback(_update_tqdm, bar_id, ordered=True, postfix_dict=postfix, s=progress)
            else:
                _ = jax.lax.cond(
                    i % print_rate == 0,
                    lambda: callback(_update_tqdm, bar_id, ordered=True, postfix_dict=postfix),
                    lambda: None,
                )
            return _carry

        carry = jax.lax.cond(
            iter_num == 0,
            _inner_init,
            _inner_update,
            iter_num,
            carry,
        )

        return carry

    def close_tqdm(result: B, iter_num: int, bar_id: int, progress: Optional[float] = None) -> B:
        def _inner_close(_result: B) -> B:
            callback(_close_tqdm, bar_id, ordered=True)
            return _result

        if schedule_mode:
            # Close when progress reaches or exceeds s_max
            pred = (progress >= total) if progress is not None else False
            result = jax.lax.cond(pred, _inner_close, lambda r: r, result)
        else:
            result = jax.lax.cond(iter_num + 1 == n, _inner_close, lambda r: r, result)
        return result

    return update_progress_bar, close_tqdm
