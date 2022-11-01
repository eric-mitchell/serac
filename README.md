## Semi-Parametric Editing with a Retrieval-Augmented Counterfactual Model

Code and data for the ICML 2022 paper *Memory-based Model Editing at Scale*.

See the paper [here](https://arxiv.org/pdf/2206.06520.pdf) and the project website [here](https://sites.google.com/view/serac-editing).

## Setup

### Environment

This codebase uses Python 3.7.9. Other versions may work as well.

Create a virtualenv ([pyenv](https://github.com/pyenv/pyenv) can help with this)
and install the dependencies:

    $ python -m venv env
    $ source env/bin/activate
    (env) $ pip install -r requirements.txt

### Data

You can download the data needed for this project from
[this Google Drive link](https://drive.google.com/file/d/1W-7Yb0eMxwZqdr7aeSgvZnbFKkzwavn6/view?usp=sharing).
You just need to unzip the archive into the top-level `serac` directory.

## Running the code

You can run the code with:

    (env) $ python -m run +alg=ALG +experiment=EXP +model=MODEL
    
See the `scripts/` directory for examples. `ALG` may be one of:
- rep [SERAC]
- gtn [MEND; [Mitchell et al., 2022](https://arxiv.org/pdf/2110.11309.pdf)]
- enn [Editable Neural Networks; [Sinitsin et al., 2020](https://arxiv.org/pdf/2004.00345.pdf)]
- lu [lookup cache baseline]
- ft [fine-tuning baseline]

The `EXP` argument may be one of:
- zsre [question-answering; must be used with `MODEL=t5large`]
- fnli [fact-checking; must be used with `MODEL=bert-base`]
- sent [sentiment editing; must be used with `MODEL=blender-small`]

## Citing the paper
If this repository is useful for your own research, you can cite our work with the following BibTeX entry:

    @inproceedings{mitchell2022memory,
        title={Memory-Based Model Editing at Scale},
        author={Mitchell, Eric and Lin, Charles and Bosselut, Antoine and Finn, Chelsea and Manning, Christopher D.},
        booktitle={International Conference on Machine Learning},
        url={https://arxiv.org/pdf/2206.06520.pdf},
        year={2022},
    }  
