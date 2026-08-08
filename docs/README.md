# Documentation

The docs are available on [![Documentation Status](https://readthedocs.org/projects/pcx/badge/?version=latest)](https://pcx.readthedocs.io/en/latest/?badge=latest)

The docs are built with `Sphinx` from the `/docs` folder. Read the Docs rebuilds and publishes them
automatically on every push to `main`; see [.readthedocs.yaml](../.readthedocs.yaml) for that pipeline.

## Build locally

From the repository root:

```shell
just docs
```

That installs the `docs` dependency group, mirrors the tutorial notebooks into the docs tree, refreshes the
API stubs and writes HTML to `docs/_build/html/index.html`.

The recipe is three steps, which you can also run by hand:

```shell
cp -r examples/* docs/source/examples/                       # mirror the newest tutorial notebooks
uv run --group docs sphinx-apidoc -f -o docs/source/ pcx/    # only needed when modules are added
uv run --group docs sphinx-build -b html docs docs/_build/html
```

If something doesn't work, raise an issue with Cornelius (cemde on gh).

# How to document

Sphinx uses RestructuredText not Markdown, but do not despair, it's easy to learn.

Here a few tips.

## Parameters

Use Google Style to set up parameters in docstrings. If the explanation is longer, indent:

```
    arg1 (str): Blablah
        blah
    arg2 (str): blub
```

The `Napoleon` extension of sphinx is responsible for reading the docstring parameters:

See [here](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#module-sphinx.ext.napoleon)

## Inline Code

Inside a docstring use this to leave inline code:

```
Just run this ``jax.lax.scan(fn, ...).`` to build AGI.
```

## Code Block

Inside a docstring use this:

```
    """
    Example:

    .. code-block:: python

        def f(x, count):
            count = count + x
            return (count + x,), None

        Scan(f, xs=jax.numpy.arange(5))(0)  # [0, 1, 3, 6, 10], None
    """
```

> :warning: WARNING: Just like python, restructured text is indentation-sensitive.

## Warning

To leave a warning

```
    .. warning::
        Linear regression is the best model ever.
```

## References

To reference a class use:

```
:class:`pcax.functional.Scan`
```

and for functions use

```
:func:`pcax.functional.scan`
```

## Maths

For maths use:

```
:math:`\int x^2 dx`
```

## More

Check out [this documentation](https://sublime-and-sphinx-guide.readthedocs.io/en/latest/lists.html).
