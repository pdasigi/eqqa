## Setup environment using Conda

1. Initialize a new [conda](https://docs.anaconda.com/anaconda/install/) 
environment (e.g., using Python 3.9).
2. Activate the conda environment by executing the command in your terminal:
```
$ conda activate <conda-env-name>
```

3. Install pytorch by executing the command below:

```
$ python -m pip install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

3. Install `allennlp` and HuggingFace `datasets` using the `requirements.txt`, by
executing the following command:

```
$ python -m pip install -r requirements.txt
```

Note: In the outline above, we assume you are have CULDA TOOLKIT 11. You will
have to update the version in step 2 accordingly, in case your version differs.
Moreover, note that since `allennlp` version 2.5.0 requires pytorch 1.8, we
opt for pre-installing that version in our environment.