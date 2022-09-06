# privacy-glue

This repository documents PrivacyGLUE; a NLP benchmark consisting of legal-privacy related tasks.

## Dependencies :mag:

1. This repository's code was tested with Python version `3.8.13`. To sync dependencies, we recommend creating a virtual environment and installing the relevant packages with `poetry`:

    ```
    $ poetry install
    ```

    Alternatively, install dependencies in the virtual environment using `pip`:
    ```
    $ pip install -r requirements-dev.txt
    ```

    **Note:** Our `torch==1.10.0` dependency works out-of-the-box with CUDA version `10.2`. If you have a different version of CUDA, refer to the official [PyTorch](https://pytorch.org/get-started/locally/) webpage for alternative pip installation commands which will provide torch optimized for your CUDA version.

2. **Optional:** If you intend to develop this repository further, we recommend installing [`pre-commit`](https://github.com/pre-commit/pre-commit) to utilize local pre-commit hooks for various code-checks.

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

| Task             | Type                                              | Study                                                                                |
|------------------|---------------------------------------------------|--------------------------------------------------------------------------------------|
| OPP-115          | Multi-label<sup>\*</sup> sequence classification  | [Wilson et al. (2016)](https://usableprivacy.org/data) and [Mousavi et al. (2020)](https://github.com/SmartDataAnalytics/Polisis_Benchmark)                               |
| PI-Extract        | Multi-label<sup>\*</sup> sequence tagging         | [Duc et al. (2021)](https://github.com/um-rtcl/piextract_dataset)                    |
| Policy-Detection | Binary sequence classification                    | [Amos et al. (2021)](https://privacypolicies.cs.princeton.edu/)                      |
| PolicyIE-A       | Multi-class<sup>\**</sup> sequence classification | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         |
| PolicyIE-B       | Multi-label<sup>\*</sup> sequence tagging         | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         |
| PolicyQA         | Reading comprehension                             | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyQA)                         |
| PrivacyQA        | Binary sequence classification                    | [Ravichander et al. (2019)](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) |

<sup>\*</sup>Multi-label implies that each classification task can have more than one gold standard label

<sup>\*\*</sup>Multi-class implies that each classification task can only have one gold standard label out of multiple choices


## Test :microscope:

1. To run unit and integration tests, execute:

    ```
    $ pytest
    ```

2. To run a `mypy` type-integrity test, execute:

    ```
    $ mypy
    ```
