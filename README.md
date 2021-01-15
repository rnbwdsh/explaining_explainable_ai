# Exercise: Explaining an AI Model used for explaining similarity between Explainable AI exercises


**Student:** Markus Vogl<br>
**Matr. Nr:** k1155575<br>
**Course:** Explainable AI<br>

![](https://i.imgflip.com/4tyf70.jpg)

## Data: Exercise data
The submitted ipython notebooks of this course are used.

* **Fetching:** All projects are pulled in parallel via the system git that requires the correct credentials on your system in your ~/.gitconfig directory.

* **Filtering:** I use jupyter-nbconvert to extract the source code from ipython files (the typical hand-in-format) from the root of the project.

* **Preprocessing:** As the model is limited to 512 tokens (not signs, source code is tokenzied), I strip all comments, emtpy newlines and outputs via regex.

## Fun fact: Data sizes

![](https://s12.directupload.net/images/210115/6v57kc5w.png)

## Model: CodeBERT
The used model is Microsofts [codeBERT](https://github.com/microsoft/CodeBERT), a variation of **RoBERTa** pre-trained on programming languages. It's based on the huggingface transformers libary which itself is based on pytorch.

The sequences have to be trimmed to a length of 512, as [this has been proven to be easy and effective](https://stackoverflow.com/questions/58636587/how-to-use-bert-for-long-text-classification). Even though there exist other approaches that work better for some scenarios.

* BERT is currently the state of the art for text encoding / embedding
* Pretrained on normal human text -> CodeBert
* I chose CodeBert over CuBERT, because Microsoft seems to be a better source

## Explainability model: Embedding Projector
The explainability model is the paper [Embedding Projector: Interactive Visualization and Interpretation of Embeddings](https://arxiv.org/abs/1611.05469) by Smilkov, et al. from 2016.

![](https://raw.githubusercontent.com/justindujardin/projector/ab50806029db3f7dea9d3064ce5ac7cb64fc7753/oss_data/spaCy_attn_learned_query_vector_preview.gif)

## Explainability framework: Hohman et al
Short exploration of the big questions posed in the format [Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers](https://arxiv.org/abs/1801.06889) by Hohman et. al from 2018.

![](https://fredhohman.com/visual-analytics-in-deep-learning/images/deepvis.png)

## Why, Who, What, When, Where, How
According to Hohnman et. al the *Embedding Projector* paper already covers:

Question | Criterion | Explanation
--- | --- | ---
Why | Interpretability & Explainability | You can interpret and explain embeddings of any kind
Who | Model Users | Data Scientists
What | Individual Computational Units | The tool can downproject with PCA and TSNE
What | Neurons in High-dimensional Space | Embeddings
What | Aggregated Information | You can aggregate multiple embeddings and compare them
When | After Training |
Where | NIPS Conference |
How | Dimensionality Reduction & Scatter Plots | PCA and TSN-E
How | Instance-based Analysis & Exploration | Exploring similiar

## Contributions

In addition to the stated factors, this adds the explainability approaches:

Question | Criterion | Explanation
--- | --- | ---
Why | Comparing & Selecting Models | This allows you to plug in other encoders like LSTMs, other BERT's etc. to compare them easily
Why | Education | It's an easy showcase of git, BERT embedding-extraction, visualization
Why | Education | It's meant to compare student exercises and find plagiarism
Who | Model Developers & Builders | as the students in this course can compare their approach to the other teams
Who | Model users | Teachers of any github classroom can just plug in their data and start
Who | Non-experts | can just plug in their github classroom data, run it and get cool visualizations (given they manage the setup)
Where | - | JKU Linz, XAI Course
How | Interactive Experimentation | You can change parameters like stripping newlines/comments and see how that effects your embeddings

## Data


```python
classroom = "jku-icg-classroom"
prefix = "xai_proj_space_2020"
teams = ['xai',
 'xai-wyoming',
 'backpropagatedstudents',
 'aikraken',
 'mysterious-potatoes',
 'the-explainables',
 'aiexplained',
 'group0',
 'xai-explainable-black-magic',
 'viennxai',
 'xai_group_a',
 'hands-on-xai',
 'xai-random-group',
 'feel_free_2_join',
 'forum_feel_free_to_join',
 'explain_it_explainable',
 'yet-another-group',
 'explanation-is-all-you-need',
 'let_me_explain_you',
 'dirty-mike-and-the-gang',
 'explain-the-unexplainable',
 'nothin_but_a_peanut',
 '3_and_1-2_ger']
```

## Fetching from github


```python
import re, os, itertools, pandas
from multiprocessing import Pool
# also requires the system utilities git, rm and jupyter-nbconvert
EXT = "ipynb"
COLUMNS = ["filename", "team", "exercise", "url"]

# Fetch contents of ipynb files
def fetch(team, strip_comments=True, strip_empty_lines=True):
    path = f"{classroom}/{prefix}/{team}"
    cp = f"{classroom}/{prefix}"
    if not os.path.exists(path):
        os.system(f"git clone git@github.com:{cp}-{team}.git {path}")
    print("⬇️", end="")
    file_content = {}
    files = filter(lambda n: n.endswith(EXT), os.listdir(path))
    for notebook in files:
        full_url = f"https://github.com/{cp}-{team}/{notebook}"
        cmd = f"jupyter-nbconvert {path}/{notebook} --to python --stdout"
        fc = os.popen(cmd).read()
        if strip_comments:    fc = re.sub("#.+", "", fc)
        if strip_empty_lines: fc = re.sub("\n+", "\n", fc)
        no_ext = (notebook.replace('.'+EXT, '')
        file_content[no_ext, team, prefix, full_url)] = fc
    print("✅", end="")
    return file_content
```

## Multithreading!


```python
def fetch_multithreaded():
    pool = Pool(len(teams))
    dicts = pool.map(fetch, teams)
    items = [fc.items() for fc in dicts]
    items_flat = itertools.chain(*items)
    return dict(items_flat)

file_content = fetch_multithreaded(); print()
prefix = "xai_model_explanation_2020"
file_content.update(fetch_multithreaded())
pandas.DataFrame(file_content.keys(), columns=COLUMNS)
```

    ⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    ⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️⬇️✅⬇️⬇️⬇️⬇️⬇️⬇️✅✅⬇️⬇️⬇️⬇️⬇️⬇️⬇️✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>filename</th>
      <th>team</th>
      <th>exercise</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>solution</td>
      <td>xai</td>
      <td>xai_proj_space_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_proj_...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>solution</td>
      <td>xai-wyoming</td>
      <td>xai_proj_space_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_proj_...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>solution</td>
      <td>backpropagatedstudents</td>
      <td>xai_proj_space_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_proj_...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>solution</td>
      <td>aikraken</td>
      <td>xai_proj_space_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_proj_...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>solution</td>
      <td>mysterious-potatoes</td>
      <td>xai_proj_space_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_proj_...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>6 - LIME_explanations_run</td>
      <td>nothin_but_a_peanut</td>
      <td>xai_model_explanation_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_model...</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Untitled</td>
      <td>nothin_but_a_peanut</td>
      <td>xai_model_explanation_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_model...</td>
    </tr>
    <tr>
      <th>75</th>
      <td>imagenet labels to pkl file</td>
      <td>nothin_but_a_peanut</td>
      <td>xai_model_explanation_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_model...</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2 - Visualization of Fully Connected Layer Neu...</td>
      <td>nothin_but_a_peanut</td>
      <td>xai_model_explanation_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_model...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>solution</td>
      <td>3_and_1-2_ger</td>
      <td>xai_model_explanation_2020</td>
      <td>https://github.com/jku-icg-classroom/xai_model...</td>
    </tr>
  </tbody>
</table>
<p>78 rows × 4 columns</p>
</div>



## Embedding extraction


```python
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.eval()  # speeds up stuff

def embed(file_content, tokenizer, model, max_length=512):
    content_list = list(file_content.values())
    # as pytorch, with padding, truncated to exactly 512 length
    tokens = tokenizer(content_list, return_tensors="pt", padding=True, 
                       max_length=max_length, truncation=True)
    # Return dict of {filename : numpy bert token}
    return model(**tokens)["pooler_output"].detach().numpy()

embedding = embed(file_content, tokenizer, model)

# save files for visualizer
np.savetxt(classroom+"-embedding.tsv", embedding, delimiter="\t")
names = ["\t".join(fc) for fc in file_content]
hdr = "filename\tteam\tproject\tfull_url"
np.savetxt(classroom+"-names.tsv", names, fmt="%s", header="\t".join(COLUMNS))
```

## Visualizer
Standalone instances:
* https://projector.tensorflow.org/
* https://justindujardin.github.io/projector/ (Works better for me for some reason)

Code:
* https://github.com/justindujardin/projector

# Visualization: Code similarity betwen projects

![](https://s12.directupload.net/images/210115/qisi9nmh.png)

# Visualization: Code similarity between teams

Example: Two teams left in the given example - distance 0.0 as it's the same file.

![](https://s12.directupload.net/images/210115/qlj38o6e.png)
