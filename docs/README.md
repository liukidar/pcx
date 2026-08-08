# Documentation

The docs are available on [![Documentation Status](https://readthedocs.org/projects/pcx/badge/?version=latest)](https://pcx.readthedocs.io/en/latest/?badge=latest)

## Contribute

You need to compile docs yourself using `Sphinx`. Once, closer to deployment this will be automated.

The docs are in the `/docs` folder. You need the `sphinx` related requirements from the poetry `dev` group to build it.

## Build

To build the documentation, do the following.

1. Activate an environment where `pcax` is installed. The script here assumes that `import pcax` works.

2. Open docs folder

```
cd /path/to/repo/docs
```

3. Copy the newest version of the example notebooks:

```
cp ../examples/* source/examples
```

4. When new modules are added, and not just updated, run this command:

```
sphinx-apidoc -f -o ../docs/source/ ../pcax/
```

5. To compile the docs run:

```
make html
```

Now you should have them in `/docs/_build/html/index.html` to launch from browser.

## Outlook

> :warning: WARNING: Right now you have to compile them yourself everytime you change something.

Once, we are closer to release, the docs will be automatically compiled everytime git `main` is updated and releases are created and uploaded to a webserver.

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
