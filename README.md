# privacy-glue

This repository documents PrivacyGLUE; a NLP benchmark consisting of legal-privacy related tasks.

## Dependencies :mag:

This repository's code was tested with Python version `3.8.12`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages via `pip`:

```
$ pip install -r requirements.txt
```

**Note:** Our `torch==1.10.0` dependency works out-of-the-box with CUDA version `10.2`. If you have a different version of CUDA, refer to the official [PyTorch](https://pytorch.org/get-started/locally/) webpage for alternative pip installation commands which will provide torch optimized for your CUDA version.

## Initialization :fire:

1. To clone and set up necessary Git submodules, simply execute:

    ```
    $ bash scripts/prepare_submodules.sh
    ```

2. To download and prepare necessary data, simply execute:

    ```
    $ bash scripts/prepare_data.sh
    ```

3. **Optional:** Initialize git hooks to manage development workflows such as linting shell scripts and keeping python dependencies up-to-date:

    ```
    $ bash scripts/prepare_git_hooks.sh
    ```

## Test :microscope:

1. To run unit and integration tests, execute:

    ```
    $ pytest
    ```

2. To run a `mypy` type-integrity test, execute:

    ```
    $ mypy
    ```
