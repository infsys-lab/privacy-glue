# PrivacyGLUE: A Benchmark Dataset for General Language Understanding in Privacy Policies

<p align="center">
<img src="./data/assets/privacy_glue_flowchart.png">
</p>

This repository functions as the official codebase for the *"PrivacyGLUE: A Benchmark Dataset for General Language Understanding in Privacy Policies"* [paper](https://www.mdpi.com/2076-3417/13/6/3701) published in the MDPI Applied Sciences special issue for NLP and applications.

PrivacyGLUE is the first comprehensive privacy-oriented NLP benchmark comprising 7 relevant and high-quality privacy tasks for measuring general language understanding in the privacy language domain. We release performances from the BERT, RoBERTa, Legal-BERT, Legal-RoBERTa and PrivBERT pretrained language models and perform model-pair agreement analysis to detect examples where models benefited from domain specialization. Our findings show that PrivBERT, the only model pretrained on privacy policies, outperforms other models by an average of 2–3% over all PrivacyGLUE tasks, shedding light on the importance of in-domain pretraining for privacy policies.

*Note that a previous version of this paper was [submitted](https://openreview.net/forum?id=n3xGexO17SJ) to the ACL Rolling Review (ARR) on 16th December 2022 before resubmission to the MDPI Applied Sciences special issue on NLP and applications on 3rd February 2023.*

## Table of Contents

1. [Tasks](#Tasks-runner)
2. [Leaderboard](#Leaderboard-checkered_flag)
3. [Dependencies](#Dependencies-mag)
3. [Initialization](#Initialization-fire)
4. [Usage](#Usage-snowflake)
5. [Notebooks](#Notebooks-book)
5. [Test](#Test-microscope)
5. [Citation](#Citation-classical_building)

## Tasks :runner:

| Task             | Study                                                                                | Type                                | Train/Dev/Test Instances | Classes                |
|:-----------------|:-------------------------------------------------------------------------------------|:------------------------------------|:------------------------:|:----------------------:|
| OPP-115          | [Wilson et al. (2016)](https://usableprivacy.org/data)<sup>\*</sup>                  | Multi-label sequence classification | 2,185/550/697            | 12                     |
| PI-Extract       | [Duc et al. (2021)](https://github.com/um-rtcl/piextract_dataset)                    | Multi-task token classification     | 2,579/456/1,029          | 3/3/3/3<sup>\*\*</sup> |
| Policy-Detection | [Amos et al. (2021)](https://privacypolicies.cs.princeton.edu/)                      | Binary sequence classification      | 773/137/391              | 2                      |
| PolicyIE-A       | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         | Multi-class sequence classification | 4,109/100/1,041          | 5                      |
| PolicyIE-B       | [Ahmad et al. (2021)](https://github.com/wasiahmad/PolicyIE)                         | Multi-task token classification     | 4,109/100/1,041          | 29/9<sup>\*\*</sup>    |
| PolicyQA         | [Ahmad et al. (2020)](https://github.com/wasiahmad/PolicyQA)                         | Reading comprehension               | 17,056/3,809/4,152       | --                     |
| PrivacyQA        | [Ravichander et al. (2019)](https://github.com/AbhilashaRavichander/PrivacyQA_EMNLP) | Binary sequence classification      | 157,420/27,780/62,150    | 2                      |

<sup>\*</sup>Data splits were not defined in Wilson et al. (2016) and were instead taken from [Mousavi et al. (2020)](https://github.com/SmartDataAnalytics/Polisis_Benchmark)

<sup>\*\*</sup>PI-Extract and PolicyIE-B consist of four and two subtasks respectively, and the number of BIO token classes per subtask are separated by a forward-slash character.

## Leaderboard :checkered_flag:

Our current leaderboard consists of the BERT ([Devlin et al., 2019](https://aclanthology.org/N19-1423/)), RoBERTa ([Liu et al., 2021](https://aclanthology.org/2021.ccl-1.108/)), Legal-BERT ([Chalkidis et al., 2020](https://aclanthology.org/2020.findings-emnlp.261/)), Legal-RoBERTa ([Geng et al., 2021](https://arxiv.org/abs/2109.06862)) and PrivBERT ([Srinath et al., 2021](https://aclanthology.org/2021.acl-long.532/)) models.

| Task             | Metric<sup>\*</sup> | BERT                | RoBERTa             | Legal-BERT          | Legal-RoBERTa       | PrivBERT                |
|:-----------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:------------------------|
| OPP-115          | m-F<sub>1</sub>     | 78.4<sub>±0.6</sub> | 79.5<sub>±1.1</sub> | 79.6<sub>±1.0</sub> | 79.1<sub>±0.7</sub> | **82.1**<sub>±0.5</sub> |
|                  | µ-F<sub>1</sub>     | 84.0<sub>±0.5</sub> | 85.4<sub>±0.5</sub> | 84.3<sub>±0.7</sub> | 84.7<sub>±0.3</sub> | **87.2**<sub>±0.4</sub> |
| PI-Extract       | m-F<sub>1</sub>     | 60.0<sub>±2.7</sub> | 62.4<sub>±4.4</sub> | 59.5<sub>±3.0</sub> | 60.5<sub>±3.9</sub> | **66.4**<sub>±3.4</sub> |
|                  | µ-F<sub>1</sub>     | 60.0<sub>±2.7</sub> | 62.4<sub>±4.4</sub> | 59.5<sub>±3.0</sub> | 60.5<sub>±3.9</sub> | **66.4**<sub>±3.4</sub> |
| Policy-Detection | m-F<sub>1</sub>     | 85.3<sub>±1.8</sub> | 86.9<sub>±1.3</sub> | 86.6<sub>±1.0</sub> | 86.4<sub>±2.0</sub> | **87.3**<sub>±1.1</sub> |
|                  | µ-F<sub>1</sub>     | 92.1<sub>±1.2</sub> | 92.7<sub>±0.8</sub> | 92.7<sub>±0.5</sub> | 92.4<sub>±1.3</sub> | **92.9**<sub>±0.8</sub> |
| PolicyIE-A       | m-F<sub>1</sub>     | 72.9<sub>±1.7</sub> | 73.2<sub>±1.6</sub> | 73.2<sub>±1.5</sub> | 73.5<sub>±1.5</sub> | **75.3**<sub>±2.2</sub> |
|                  | µ-F<sub>1</sub>     | 84.7<sub>±1.0</sub> | 84.8<sub>±0.6</sub> | 84.7<sub>±0.5</sub> | 84.8<sub>±0.3</sub> | **86.2**<sub>±1.0</sub> |
| PolicyIE-B       | m-F<sub>1</sub>     | 50.3<sub>±0.7</sub> | 52.8<sub>±0.6</sub> | 51.5<sub>±0.7</sub> | 53.5<sub>±0.5</sub> | **55.4**<sub>±0.7</sub> |
|                  | µ-F<sub>1</sub>     | 50.3<sub>±0.5</sub> | 54.5<sub>±0.7</sub> | 52.2<sub>±1.0</sub> | 53.6<sub>±0.9</sub> | **55.7**<sub>±1.3</sub> |
| PolicyQA         | s-F<sub>1</sub>     | 55.7<sub>±0.5</sub> | 57.4<sub>±0.4</sub> | 55.3<sub>±0.7</sub> | 56.3<sub>±0.6</sub> | **59.3**<sub>±0.5</sub> |
|                  | EM                  | 28.0<sub>±0.9</sub> | 30.0<sub>±0.5</sub> | 27.5<sub>±0.6</sub> | 28.6<sub>±0.9</sub> | **31.4**<sub>±0.6</sub> |
| PrivacyQA        | m-F<sub>1</sub>     | 53.6<sub>±0.8</sub> | 54.4<sub>±0.3</sub> | 53.6<sub>±0.8</sub> | 54.4<sub>±0.5</sub> | **55.3**<sub>±0.6</sub> |
|                  | µ-F<sub>1</sub>     | 90.0<sub>±0.1</sub> | 90.2<sub>±0.0</sub> | 90.1<sub>±0.1</sub> | 90.2<sub>±0.1</sub> | **90.2**<sub>±0.1</sub> |

<sup>\*</sup>m-F<sub>1</sub>, µ-F<sub>1</sub>, s-F<sub>1</sub> and EM refer to the Macro-F<sub>1</sub>, Micro-F<sub>1</sub>, Sample-F<sub>1</sub> and Exact Match metrics respectively

## Dependencies :mag:

1. This repository was tested against Python version `3.8.13` and CUDA version `11.7`. Create a virtual environment with the same python version and install dependencies with [`poetry`](https://python-poetry.org/):

    ```
    $ poetry install
    ```

    Alternatively, install dependencies in the virtual environment using `pip`:
    ```
    $ pip install -r requirements.txt
    ```

2.  Install Git [`LFS`](https://git-lfs.github.com/) to access upstream task data. We utilized version `3.2.0` in our implementation.

3. **Optional:** To further develop this repository, install [`pre-commit`](https://github.com/pre-commit/pre-commit) to setup pre-commit hooks for code-checks.

## Initialization :fire:

1. To prepare git submodules and data, execute:

    ```
    $ bash scripts/prepare.sh
    ```

2. **Optional:** To install pre-commit hooks for further development of this repository, execute:

    ```
    $ pre-commit install
    ```

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

**Note**: Replace the `<device_id>` argument with a GPU ID or comma-separated GPU IDs to run single-GPU or multi-GPU fine-tuning respectively. Correspondingly, replace the `<model>` argument with one of our supported models listed in the usage documentation above.

## Notebooks :book:

We utilize the following `ipynb` notebooks for analyses outside of the PrivacyGLUE benchmark:

| Notebook                                                                         | Description                                                                                           |
|:---------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------|
| [visualize_domain_embeddings.ipynb](notebooks/visualize_domain_embeddings.ipynb) | Compute and visualize BERT embeddings for Wikipedia, EURLEX and Privacy Policies using t-SNE and UMAP |
| [visualize_results.ipynb](notebooks/visualize_results.ipynb)                     | Plot benchmark results and perform significance testing                                               |
| [inspect_predictions.ipynb](notebooks/inspect_predictions.ipynb)                 | Inspect test-set predictions for model-pair agreement analysis                                        |

## Test :microscope:

1. To run unit tests, execute:

    ```
    $ make test
    ```

2. To run integration tests, execute:

    ```
    $ CUDA_VISIBLE_DEVICES=<device_id> make integration
    ```

    **Note:** Replace the `<device_id>` argument with a GPU ID or comma-separated GPU IDs to run single-GPU or multi-GPU integration tests respectively. Alternatively, pass an empty string to run CPU integration tests.

## Citation :classical_building:

If you found PrivacyGLUE useful, we kindly ask you to cite our paper as follows:

```bibtex
@Article{app13063701,
  AUTHOR =       {Shankar, Atreya and Waldis, Andreas and Bless, Christof and
                  Andueza Rodriguez, Maria and Mazzola, Luca},
  TITLE =        {PrivacyGLUE: A Benchmark Dataset for General Language
                  Understanding in Privacy Policies},
  JOURNAL =      {Applied Sciences},
  VOLUME =       {13},
  YEAR =         {2023},
  NUMBER =       {6},
  ARTICLE-NUMBER ={3701},
  URL =          {https://www.mdpi.com/2076-3417/13/6/3701},
  ISSN =         {2076-3417},
  ABSTRACT =     {Benchmarks for general language understanding have been
                  rapidly developing in recent years of NLP research,
                  particularly because of their utility in choosing
                  strong-performing models for practical downstream
                  applications. While benchmarks have been proposed in the legal
                  language domain, virtually no such benchmarks exist for
                  privacy policies despite their increasing importance in modern
                  digital life. This could be explained by privacy policies
                  falling under the legal language domain, but we find evidence
                  to the contrary that motivates a separate benchmark for
                  privacy policies. Consequently, we propose PrivacyGLUE as the
                  first comprehensive benchmark of relevant and high-quality
                  privacy tasks for measuring general language understanding in
                  the privacy language domain. Furthermore, we release
                  performances from multiple transformer language models and
                  perform model&ndash;pair agreement analysis to detect tasks
                  where models benefited from domain specialization. Our
                  findings show the importance of in-domain pretraining for
                  privacy policies. We believe PrivacyGLUE can accelerate NLP
                  research and improve general language understanding for humans
                  and AI algorithms in the privacy language domain, thus
                  supporting the adoption and acceptance rates of solutions
                  based on it.},
  DOI =          {10.3390/app13063701}
}
```
