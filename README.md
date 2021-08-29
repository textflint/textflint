<p align="center"><img src="images/logo.png" alt="Textflint Logo" height="100"></p>

<h3 align="center">Unified Multilingual Robustness Evaluation Toolkit 
  for Natural Language Processing</h3>
<p align="center">
  <a>
    <img src="https://github.com/textflint/textflint/actions/workflows/python-package.yml/badge.svg" alt="Github Runner Covergae Status">
  </a>

  <a href="https://www.textflint.io/textflint">
  	<img alt="Website" src="https://img.shields.io/website?up_message=online&url=https%3A%2F%2Fwww.textflint.io%2F">
  </a>

  <a>
  	<img alt="License" src="https://img.shields.io/badge/license-GPL%20v3-brightgreen">
  </a>

  <a href="https://badge.fury.io/py/textflint">
  	<img alt="GitHub release (latest by date)" 	src="https://img.shields.io/github/v/release/textflint/textflint?label=release">
  </a>
</p>


TextFlint is a multilingual robustness evaluation platform for natural language processing, which unifies text **transformation**, **sub-population**, **adversarial attack**,and their combinations to provide a comprehensive robustness analysis. So far, TextFlint supports 13 NLP tasks.

> If you're looking for robustness evaluation results of SOTA models, you might want the [TextFlint IO](https://www.textflint.io/textflint) page.

## Features

- **Full coverage of transformation types**, including 20 general transformations, 8 subpopulations and 60 task-specific transformations, as well as thousands of their combinations.
- **Subpopulation**, which is to identify the specific part of dataset on which the target model performs poorly. 
- **Adversarial attack** aims to find a perturbation of an input text that is able to fool the given model.
- **Complete analytical report** to accurately explain where your model's shortcomings are, such as the problems in lexical rules or syntactic rules. 

## Online Demo

You can test most of transformations directly on our [online demo](https://www.textflint.io/tutorials). 

## Table of Contents

- [Setup](#setup)
- [Usage](#usage)
- [Architecture](#Architecture)
- [Learn More](#learn-more)
- [Contributing](#contributing)
- [Citation](#Citation)

## Setup

Require **python version >= 3.7**, recommend install with `pip`.

```shell
pip install textflint
```

Once TextFlint is installed, you can run it via command-line (`textflint ...`) or integrate it inside another NLP project.

## Usage

### Workflow



<img src="images/workflow.png" style="zoom:50%;" />

The general workflow of TextFlint is displayed above. Evaluation of target models could be divided into three steps:

1. For input preparation, the original dataset for testing, which is to be loaded by `Dataset`, should be firstly formatted as a series of `JSON` objects. You can use the built-in `Dataset` following this [instruction](docs/user/components/4_Sample_Dataset.ipynb). TextFlint configuration is specified by `Config`. Target model is also loaded as `FlintModel`.
2. In adversarial sample generation, multi-perspective transformations (i.e., [80+Transformation](docs/user/components/transformation.md), [Subpopulation](docs/user/components/subpopulation.md) and [AttackRecipe](https://github.com/QData/TextAttack)), are performed on `Dataset` to generate transformed samples. Besides, to ensure semantic and grammatical correctness of transformed samples, [Validator](docs/user/components/validator.md) calculates confidence of each sample to filter out unacceptable samples.
3. Lastly, `Analyzer` collects evaluation results and `ReportGenerator` automatically generates a comprehensive report of model robustness. 

For example, on the Sentiment Analysis (SA) task, this is a statistical chart of the performance of`XLNET`  with different types of `Transformation`/`Subpopulation`/`AttackRecipe` on the `IMDB` dataset. 

<img src="images/report.png" alt="" style="zoom:100%" />

We release tutorials of performing the whole pipeline of TextFlint on various tasks, including:

* [Machine Reading Comprehension](docs/user/tutorials/9_MRC.ipynb)
* [Part-of-speech Tagging](docs/user/tutorials/7_BERT%20for%20POS%20tagging.ipynb)
* [Named Entity Recognition](docs/user/tutorials/11_NER.ipynb)
* [Chinese Word Segmentation](docs/user/tutorials/10_CWS.ipynb)

### Quick Start

Using TextFlint to verify the robustness of a specific model is as simple as running the following command:

```shell
$ textflint --dataset input_file --config config.json
```

where *input\_file* is the input file of csv or json format, *config.json* is a configuration file with generation and target model options.  Transformed datasets would save to your out dir according to your *config.json*. 

Based on the design of decoupling sample generation and model verification, **TextFlint** can be used inside another NLP project with just a few lines of code.

```python
from textflint import Engine

data_path = 'input.json'
config = 'config.json'
engine = Engine()
engine.run(data_path, config)
```

For more input and output instructions of TextFlint, please refer to the [IO format  document](docs/user/components/IOFormat.md).

## Architecture

<img src="images/architecture.png" style="zoom:50%;" />

***Input layer:*** receives textual datasets and models as input, represented as `Dataset` and `FlintModel` separately.

- **`DataSet`**: a container, provides efficient and handy operation interfaces for `Sample`. `Dataset` supports loading, verification, and saving data in Json or CSV format for various NLP tasks. 
- **`FlintModel`**: a target model used in an adversarial attack.

 ***Generation layer:***  there are mainly four parts in generation layer:

- **`Subpopulation`**: generates a subset of a `DataSet`. 
- **`Transformation`**: transforms each sample of `Dataset` if it can be transformed. 
- **`AttackRecipe`**: attacks the `FlintModel` and generates a `DataSet` of adversarial examples.
- **`Validator`**: verifies the quality of samples generated by `Transformation` and `AttackRecipe`.

> textflint provides an interface to integrate the easy-to-use adversarial attack recipes implemented based on `textattack`. Users can refer to [textattack](https://github.com/QData/TextAttack) for more information about the supported `AttackRecipe`.

***Report layer:*** analyzes model testing results and provides robustness report for users.

## Learn More

| Section                                                      | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Documentation](https://textflint.readthedocs.io/)           | Full API documentation and tutorials                         |
| [Tutorial](https://github.com/textflint/textflint/tree/master/docs/user) | The tutorial of textflint components and pipeline            |
| [Website](https://www.textflint.io/textflint)                | Provides evaluation results of SOTA models and transformed data download |
| [Online Demo](https://www.textflint.io/tutorials)            | Interactive demo to try single text transformations          |
| [Paper](https://aclanthology.org/2021.acl-demo.41.pdf) | Our system paper which was received by ACL2021               |

## Contributing

We welcome community contributions to TextFlint in the form of bugfixes 🛠️ and new features💡!   If you want to contribute, please first read [our contribution guideline](CONTRIBUTING.md).

## Citation

If you are using TextFlint for your work, please kindly cite our [ACL2021 TextFlint demo paper](https://aclanthology.org/2021.acl-demo.41.pdf):

```latex
@inproceedings{wang-etal-2021-textflint,
    title = {TextFlint: Unified Multilingual Robustness Evaluation Toolkit for Natural Language Processing},
    author = {Wang, Xiao  and Liu, Qin  and Gui, Tao  and Zhang, Qi and others},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing: System Demonstrations},
    month = {aug},
    year = {2021},
    address = {Online},
    publisher = {Association for Computational Linguistics},
    url = {https://aclanthology.org/2021.acl-demo.41},
    doi = {10.18653/v1/2021.acl-demo.41},
    pages = {347--355}
}
```
