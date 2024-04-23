# Documentation

## Introduction

You need to compile docs yourself using `Sphinx`. Once, closer to deployment this will be automated.

The docs are in the `/docs` folder. You need the `sphinx` related requirements from `/requirements/requirements-dev.txt` to build it. Once, set up, it will be hosted automatically.

## Build

To build the documentation, do the following.

1. Open docs folder

```
cd /path/to/repo/docs
```

2. Copy the newest version of the example notebooks:

```
cp ../examples/* source/examples
```

3. When new modules are added, and not just updated, run this command:

```
sphinx-apidoc -f -o ../docs/source/ ../pcax/
```

4. To compile the docs run:

```
make html
```

Now you should have them in `/docs/_build/html/index.html` to launch from browser.

> :warning: WARNING: Right now you have to compile them yourself everytime you change something.

Once, we are closer to release, the docs will be automatically compiled everytime git `main` is updated and releases are created and uploaded to a webserver.

If something doesn't work, raise an issue with Cornelius (cemde on gh).
