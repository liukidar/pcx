# PCX Development

## General Guide

Please use a new branch to contribute. Once you are done writing the feature, please open a Pull Request. A few github actions will be triggered: Formatting, Linting, Type Checking and Testing. These steps ensure that the library maintains a certain level of quality and fascilitate collaboration.

-   Formatting: We use `black` to format the code. Check out [pyproject.toml](pyproject.toml) for the arguments. This ensures that all files from all authors are coherently formatted.
-   Linting: This checks the code for style quality using `flake8` and `isort`. `flake8` complains about style errors, as for example variable names that are too generic or there exist unused imports. `isort` complains if the imports within each block are not alphabetically ordered. For linting arguments check [setup.cfg](setup.cfg).
-   Type Checking: This checks if type hints match the types that are actually passed through the code using `mypy`. This isnt infallable but it helps a lot making the typing more explicit. If you can't find a solution to satisfy `mypy` you can mark that line with a `# type: ignore` comment and `mypy` will ignore errors in this line. For arguments check [pyproject.toml](pyproject.toml).
-   Testing: All code in the library should be explicitly tested. We use `pytest` to do that. Please add tests for any feature you implement.

The GHA only check that everything is in order, but does not alter the code for you. Please, do the formatting, testing etc locally. You can use the logs of the GHA to see where it is failing. Once all the GHA pass, request Luca as a reviewer for your PR. Once he approves, you should use the `Squash and Merge` functionalty to merge your feature on the main branch. `Squash and Merge` means that all your little commits on the feature branch will be bundles into one big commit and keep the commit tree tidy.

Please add comments and docstrings to your changes.

One warning: We cannot test GPU features on GHA. Please do this locally as well, even though it might not result in an error in the GHA.

### Formatting

We are currently setting up a standard. Come back later!

### Skip Github Actions

If you want to skip github actions on a single commit (e.g. an intermediate commit or comment that does not need to be checked), you can start your commit message with `[skip ci]`.
