# privacy-glue

This repository documents PrivacyGLUE; a NLP benchmark for general language understanding in the privacy language domain.

## Dependencies :mag:

1. This repository's code was tested with Python version `3.8.13` and CUDA version `11.7`. To sync dependencies, we recommend creating a virtual environment with the same python version and installing the relevant packages with [`poetry`](https://python-poetry.org/):

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

1. To prepare the necessary git submodules and data, execute:

    ```
    $ bash scripts/prepare.sh
    ```

2. **Optional:** If you intend to further develop this repository, execute the following to initialize pre-commit hooks:

    ```
    $ pre-commit install
    ```

## Tasks :runner:

| Task             | Type                                | Study                                                                                |
|:-----------------|:------------------------------------|:-------------------------------------------------------------------------------------|
| OPP-115          | Multi-label sequence classification | [Wilson et al. (2016)](https://usableprivacy.org/data)<sup>\*</sup>                  |
| PI-Extract       | Multi-task token classification     | [Duc et al. (2021)](https://github.com/um-rtcl/piextract_dataset)                    |
| Policy-Detection | Binary sequence classification      | [Amos et al. (2021)](https://privacypolicies.cs.princeton.edu/)                      |
| PolicyIE-A       | Multi-class sequence classification | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         |
| PolicyIE-B       | Multi-task token classification     | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         |
| PolicyQA         | Reading comprehension               | [Ahmad et al. (2020)](https://github.com/wasiahmad/PolicyQA)                         |
| PrivacyQA        | Binary sequence classification      | [Ravichander et al. (2019)](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) |

<sup>\*</sup>Data splits were not defined in Wilson et al. (2016) and were instead taken from [Mousavi et al. (2020)](https://github.com/SmartDataAnalytics/Polisis_Benchmark)

## Usage :snowflake:

We use the `run_privacy_glue.sh` script to run PrivacyGLUE benchmark experiments:

```
usage: run_privacy_glue.sh [option...]

optional arguments:
  --cuda_visible_devices       <str>
                               comma separated string of integers passed
                               directly to the "CUDA_VISIBLE_DEVICES"
                               environment variable
                               (default: 0)

  --fp16                       enable 16-bit mixed precision computation
                               through NVIDIA Apex for training
                               (default: False)

  --model_name_or_path         <str>
                               model to be used for fine-tuning. Currently only
                               the following are supported:
                               "bert-base-uncased",
                               "roberta-base",
                               "nlpaueb/legal-bert-base-uncased",
                               "saibo/legal-roberta-base",
                               "mukund/privbert"
                               (default: bert-base-uncased)

  --no_cuda                    disable CUDA even when available (default: False)

  --overwrite_cache            overwrite caches used in preprocessing
                               (default: False)

  --overwrite_output_dir       overwrite run directories and saved checkpoint(s)
                               (default: False)

  --preprocessing_num_workers  <int>
                               number of workers to be used for preprocessing
                               (default: None)

  --task                       <str>
                               task to be worked on. The following values are
                               accepted: "opp_115", "piextract",
                               "policy_detection", "policy_ie_a", "policy_ie_b",
                               "policy_qa", "privacy_qa", "all"
                               (default: all)

  --wandb                      log metrics and results to wandb
                               (default: False)

  -h, --help                   show this help message and exit
```

To run the PrivacyGLUE benchmark for a supported model against all tasks, execute:

```
$ bash scripts/run_privacy_glue.sh --cuda_visible_devices <device_id> \
                                   --model_name_or_path <model> \
                                   --fp16
```

## Notebooks :book:

We utilize the following `ipynb` notebooks for general analyses outside of the PrivacyGLUE benchmark:

| Notebook                                                                         | Description                                                                                           |
|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| [visualize_domain_embeddings.ipynb](notebooks/visualize_domain_embeddings.ipynb) | Compute and visualize BERT embeddings for Wikipedia, EURLEX and Privacy Policies using t-SNE and UMAP |
| [visualize_results.ipynb](notebooks/visualize_results.ipynb)                     | Plot benchmark results and perform significance testing                                               |
| [inspect_predictions.ipynb](notebooks/inspect_predictions.ipynb)                 | Inspect test-set predictions for model agreement analysis                                             |

## Test :microscope:

1. To run unit tests, execute:

    ```
    $ make test
    ```

2. To run integration tests, execute:

    ```
    $ CUDA_VISIBLE_DEVICES=<device_id> make integration
    ```

    **Note**: Replace the `<device_id>` argument with a GPU-ID to run single-GPU integration tests or GPU-IDs to run multi-GPU integration tests. Alternatively, pass an empty string to run CPU integration tests.


## Citation :classical_building:

If you found PrivacyGLUE useful, we kindly ask you to cite our paper as follows:

```bibtex
@article{shankar2023privacyglue,
  title =        {PrivacyGLUE: A Benchmark Dataset for General Language
                  Understanding in Privacy Policies},
  author =       {Shankar, Atreya and Waldis, Andreas and Bless, Christof and
                  Rodriguez, Maria A and Mazzola, Luca},
  year =         {2023},
  publisher =    {Preprints}
}
```
