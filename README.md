# -- FORK --

## Progress bars with jax-tqdm

This project vendors a customized `jax_tqdm` with enhanced progress bars for JAX loops.

### while_tqdm

`while_tqdm` is a decorator factory that adds a tqdm progress bar to a `jax.lax.while_loop` body function.

```python
from jax_tqdm.while_tqdm import while_tqdm

@while_tqdm(
    n=None,                 # int | None; required unless using schedule mode
    print_rate=None,        # int | None; default n//20
    tqdm_type="auto",      # "auto" | "std" | "notebook"
    print_s=None,           # list/array of elapsed-time thresholds (schedule mode)
    s_max=None,             # float; required if print_s
    last_is_postfix=False,  # if True, treat carry's last element as postfix dict
    postfix_fmt_str=None,   # dict[str, str]; per-key format overrides for postfix
    desc="Training",       # forwarded to tqdm
)
def body(carry):
    i, state = carry  # i must be iteration counter when not using schedule mode
    # ... do work ...
    return (i + 1, state)
```

There are two modes:

- Iteration-based: Provide `n` (and optionally `print_rate`). The bar updates every `print_rate` iterations (defaults to about 20 updates across `n`).
- Schedule-based: Provide `print_s` (sequence of thresholds) and `s_max` (maximum elapsed-time). The bar advances only when `progress` crosses thresholds in `print_s`. In this mode, you should pass the current elapsed-time `progress` through the decorator; our wrapper handles forwarding.

### Postfix formatting

Postfix values shown next to the bar are formatted by default as follows:
- Integers: `"{value}"`
- Floats: `f"{value:.4f}"` (configurable precision via `postfix_decimals` forwarded to tqdm kwargs)

You can override the formatting per key using `postfix_fmt_str`, mapping keys to Python format specs. For example:

```python
@while_tqdm(
    n=10_000,
    postfix_fmt_str={"loss": ".3e", "acc": ".2%", "lr": ".1e"},
)
def body(carry):
    i, state, metrics = carry
    # set metrics as the last element when using last_is_postfix=True
    return (i + 1, state, metrics)
```

If a format fails for a given value, the code falls back to sensible defaults.

Identical kwargs and behavior are available for `scan_tqdm` and `loop_tqdm`.

# JAX-Tqdm

Add a [tqdm](https://github.com/tqdm/tqdm) progress bar to your JAX scans and loops.

![PyPI - Version](https://img.shields.io/pypi/v/jax-tqdm)
![PyPI - Downloads](https://img.shields.io/pypi/dm/jax-tqdm)

## Installation

Install with pip:

```bash
pip install jax-tqdm
```

## Example Usage

### In `jax.lax.scan`

```python
from jax_tqdm import scan_tqdm
from jax import lax
import jax.numpy as jnp

n = 10_000

@scan_tqdm(n)
def step(carry, x):
    return carry + 1, carry + 1

last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))
```

Where the range argument must start at 0. A tuple can be used to pass data to the scan,
as long as the first entry is a range, e.g.:

```python
from jax_tqdm import scan_tqdm
from jax import lax
import jax.numpy as jnp

n = 10
scan_data = jnp.zeros((n, 200, 100))

@scan_tqdm(n)
def step(carry, x):
    _, d = x
    return carry + 1, d

last_number, output_data = lax.scan(step, 0, (jnp.arange(n), scan_data))
```

### In `jax.lax.fori_loop`

```python
from jax_tqdm import loop_tqdm
from jax import lax

n = 10_000

@loop_tqdm(n)
def step(i, val):
    return val + 1

last_number = lax.fori_loop(0, n, step, 0)
```

where the initial loop value should start at 0.

### Scans & Loops Inside Vmap

For scans and loops inside a map, jax-tqdm can print stacked progress bars
showing the individual progress of each process. To do this you can wrap
the initial value of the loop or scan inside a `PBar` class, along with the
index of the progress bar. For example

```python
from jax_tqdm import PBar, scan_tqdm
import jax

n = 10_000

@scan_tqdm(n)
def step(carry, _):
    return carry + 1, carry + 1

def map_func(i):
    # Wrap the initial value and pass the
    # progress bar index
    init = PBar(id=i, carry=0)
    final_value, _all_numbers = jax.lax.scan(
        step, init, jax.numpy.arange(n)
    )
    return (
        final_value.carry,
        _all_numbers,
    )

last_numbers, all_numbers = jax.vmap(map_func)(jax.numpy.arange(10))
```

The indices of the progress bars should be contiguous integers starting
from 0.

### Print Rate

By default, the progress bar is updated 20 times over the course of the scan/loop
(for performance purposes, see [below](#why-jax-tqdm)). This
update rate can be manually controlled with the `print_rate` keyword argument. For
example:

```python
from jax_tqdm import scan_tqdm
from jax import lax
import jax.numpy as jnp

n = 10_000

@scan_tqdm(n, print_rate=2)
def step(carry, x):
    return carry + 1, carry + 1

last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))
```

will update every other step.

### Progress Bar Type

You can select the [tqdm](https://github.com/tqdm/tqdm) [submodule](https://github.com/tqdm/tqdm/tree/master?tab=readme-ov-file#submodules) manually with the `tqdm_type` option. The options are `'std'`, `'notebook'`, or `'auto'`.
```python
from jax_tqdm import scan_tqdm
from jax import lax
import jax.numpy as jnp

n = 10_000

@scan_tqdm(n, print_rate=1, tqdm_type='std') # tqdm_type='std' or 'notebook' or 'auto'
def step(carry, x):
    return carry + 1, carry + 1

last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))
```

### Progress Bar Options

Any additional keyword arguments are passed to the [tqdm](https://github.com/tqdm/tqdm)
progress bar constructor. For example:

```python
from jax_tqdm import scan_tqdm
from jax import lax
import jax.numpy as jnp

n = 10_000

@scan_tqdm(n, print_rate=1, desc='progress bar', position=0, leave=False)
def step(carry, x):
    return carry + 1, carry + 1

last_number, all_numbers = lax.scan(step, 0, jnp.arange(n))
```

## Why JAX-Tqdm?

JAX functions are [pure](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions),
so side effects such as printing progress when running scans and loops are not allowed.
However, the
[debug module](https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html#exploring-debug-callback)
has primitives for calling Python functions on the host from JAX code. This can be used
to update a Python tqdm progress bar regularly during the computation. JAX-tqdm
implements this for JAX scans and loops and is used by simply adding a decorator to the
body of your update function.

Note that as the tqdm progress bar is only updated 20 times during the scan or loop,
there is no performance penalty.

The code is explained in more detail in this [blog post](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/).

## Developers

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```bash
poetry install
```

### Pre-Commit Hooks

Pre commit hooks can be installed by running

```bash
pre-commit install
```

Pre-commit checks can then be run using

```bash
task lint
```

### Tests

Tests can be run with

```bash
task test
```