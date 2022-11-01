# privacy-glue

This repository documents PrivacyGLUE; a NLP benchmark consisting of legal-privacy related tasks.

## Dependencies :mag:

1. This repository's code was tested with Python version `3.8.13`. To sync dependencies, we recommend creating a virtual environment with the same python version and installing the relevant packages with `poetry`:

    ```
    $ poetry install
    ```

    Alternatively, install dependencies in the virtual environment using `pip`:
    ```
    $ pip install -r requirements.txt
    ```

2.  This repository requires a working installation of Git [`LFS`](https://git-lfs.github.com/) to access upstream task data. We utilized version `3.2.0` in our implementation.

3. **Optional:** If you intend to develop this repository further, we recommend installing [`pre-commit`](https://github.com/pre-commit/pre-commit) to utilize local pre-commit hooks for various code-checks.

## Initialization :fire:

1. To prepare the necessary git submodules and data, simply execute:

    ```
    $ bash scripts/prepare.sh
    ```

2. **Optional:** If you intend to further develop this repository, execute the following to initialize pre-commit hooks:

    ```
    $ pre-commit install
    ```

## Tasks :runner:

| Task             | Type                                               | Study                                                                                |
|------------------|----------------------------------------------------|--------------------------------------------------------------------------------------|
| OPP-115          | Multi-label<sup>\*</sup> sequence classification   | [Wilson et al. (2016)](https://usableprivacy.org/data)<sup>\*\*\*</sup>              |
| PI-Extract       | Joint multi-class<sup>\*\*</sup> sequence tagging  | [Duc et al. (2021)](https://github.com/um-rtcl/piextract_dataset)                    |
| Policy-Detection | Binary sequence classification                     | [Amos et al. (2021)](https://privacypolicies.cs.princeton.edu/)                      |
| PolicyIE-A       | Multi-class<sup>\*\*</sup> sequence classification | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         |
| PolicyIE-B       | Joint multi-class<sup>\*\*</sup> sequence tagging  | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         |
| PolicyQA         | Reading comprehension                              | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyQA)                         |
| PrivacyQA        | Binary sequence classification                     | [Ravichander et al. (2019)](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) |

<sup>\*</sup>Multi-label implies that each classification task can have more than one gold standard label

<sup>\*\*</sup>Multi-class implies that each classification task can only have one gold standard label out of multiple choices

<sup>\*\*\*</sup>Data splits were not defined in Wilson et al. (2016) and were instead taken from [Mousavi et al. (2020)](https://github.com/SmartDataAnalytics/Polisis_Benchmark)

## Test :microscope:

1. To run unit tests, execute:

    ```
    $ pytest
    ```

2. To run integration tests, execute:

    ```
    $ CUDA_VISIBLE_DEVICES=<device> pytest -m slow
    ```
    **Note**: Replace the `<device>` argument with GPU ID(s) in order to run single or multi-GPU integration tests. Alternatively, remove the `CUDA_VISIBLE_DEVICES` environment variable for CPU integration tests.

3. To run a `mypy` type-integrity test, execute:

    ```
    $ mypy
    ```
