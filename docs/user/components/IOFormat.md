## Input Format

Using TextFlint to verify the robustness of a specific model is as simple as running the following command:

```shell
$ textflint --dataset input_file --config config.json
```

where *input\_file* is the input file of csv or json format, *config.json* is a configuration file with generation and target model options. 

### Input File

*input\_file* is the input file of csv or json format. Each line of the file just contains one sample JSON. Take the input file for  **SA task** as example:

```json
{"x": "Titanic is my favorite movie.", "y": "pos", "sample_id": 0}
{"x": "I don't like the actor Tim Hill", "y": "neg", "sample_id": 1}
```

> Note that the input format of different tasks is different, please refer to this [tutorial](4_Sample_Dataset.ipynb) for details.

### Config File

*config.json* is a configuration file with generation and target model options. Take the configuration for **TextCNN** model on SA task as example:

```json
{
  "task": "SA",
  "out_dir": "./DATA/",
  "trans_methods": [
    "Ocr",
    ["InsertAdv", "SwapNamedEnt"],   
    ...
  ],
  "trans_config": {
    "Ocr": {"trans_p": 0.3},
    ...
  },
...
}
```

- *task* is the name of target task. 

- *out\_dir* is the directory where each of the generated sample and its corresponding original sample are saved.

- *flint\_model* is the python file path that saves the instance of FlintModel.

  Note that ***flint\_model*** is **not necessary** for transformation or subpopulation. You can remove this option, if you are not familar with **FlintModel**. 

- *trans\_methods* is used to specify the transformation method. For example, *"Ocr"* denotes the universal transformation **Ocr**,  and *["InsertAdv", "SwapNamedEnt"]* denotes a pipeline of task-specific transformations, namely **InsertAdv** and **SwapNamedEnt**.

- *trans\_config* configures the parameters for the transformation methods. The default parameter is also a good choice. 

## Output Format

###  Transformed Datasets

After transformation, here are the contents in `./DATA/`:

```
ori_Keyboard_2.json
ori_SwapNamedEnt_1.json
trans_Keyboard_2.json
trans_SwapNamedEnt_1.json
...
```

where the `trans_Keyboard_2.json`  contains `2` successfully transformed sample by transformation `Keyboard` and `ori_Keyboard_2.json` contains the corresponding original sample. The content in `ori_Keyboard_2.json`: 

```
{"x": "Titanic is my favorite movie.", "y": "pos", "sample_id": 0}
{"x": "I don't like the actor Tim Hill", "y": "neg", "sample_id": 1}
```

The content in `trans_Keyboard_2.json`:

```
{"x": "Titanic is my favorite m0vie.", "y": "pos", "sample_id": 0}
{"x": "I don't likR the actor Tim Hill", "y": "neg", "sample_id": 1}
```

### Robustness Report

Based on the results from Generation Layer,  TextFlint can generate three types of adversarial samples and verify the robustness of the target model. 

For example, on the Sentiment Analysis (SA) task, this is a statistical chart of the performance of`XLNET`  with different types of `Transformation`/`Subpopulation`/`AttackRecipe` on the `IMDB` dataset. 

<img src="/Users/wangxiao/code/python/flint_fix/readme/flintdoc/images/report.png" alt="" style="zoom:100%" />
