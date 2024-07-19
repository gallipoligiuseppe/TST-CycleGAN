# Self-supervised Text Style Transfer using Cycle-Consistent Adversarial Networks
This repository contains the code for the paper [Self-supervised Text Style Transfer using Cycle-Consistent Adversarial Networks](https://dl.acm.org/doi/10.1145/3678179), published in ACM Transactions on Intelligent Systems and Technology.

It includes the Python package to train and test the CycleGAN architecture for Text Style Transfer described in the paper.

## Installation
The following command will clone the project:
```
git clone https://github.com/gallipoligiuseppe/TST-CycleGAN.git
```

To install the required libraries and dependencies, you can refer to the `env.yml` file.

Before experimenting, you can create a virtual environment for the project using Conda.
```
conda create -f env.yml -n cyclegan_tst 
conda activate cyclegan_tst
```

The installation should also cover all the dependencies. If you find any missing dependency, please let us know by opening an issue.

## Usage
The package provides the scripts to implement, train and test the CycleGAN architecture for Text Style Transfer described in the paper.

Specifically, we focus on *formality* (informal ↔ formal) and *sentiment* (negative ↔ positive) transfer tasks.

## Data
### Formality transfer
According to the dataset license, you can request access to the [GYAFC](https://aclanthology.org/N18-1012/) dataset following the steps described in its official [repository](https://github.com/raosudha89/GYAFC-corpus).

Once you have gained access, put it into the `family_relationships` and `entertainment_music` directories for the *Family & Relationships* and *Entertainment & Music* domains, respectively, under the `data/GYAFC` folder. Please name the files as `[train|dev|test].[0|1].txt`, where 0 is for informal style and 1 is for formal style.

We could provide access to *mixed-style* data we use in our work after gaining access to the GYAFC dataset and verifying the dataset license.

### Sentiment transfer
We use the [Yelp](https://papers.nips.cc/paper_files/paper/2017/hash/2d2c8394e31101a261abf1784302bf75-Abstract.html) dataset following the same splits as in [Li et al.](https://aclanthology.org/N18-1169/) available in the official [repository](https://github.com/lijuncen/Sentiment-and-Style-Transfer). Put it into the `data/yelp` folder and please name the files as `[train|dev|test].[0|1].txt`, where 0 is for negative sentiment and 1 is for positive sentiment.

## Training
You can train the proposed CycleGAN architecture for Text Style Transfer using the `train.py` script. It can be customized using several command line arguments such as:
- style_a/style_b: style A/B (i.e., informal/formal or negative/positive)
- generator_model_tag: tag or path of the generator model
- discriminator_model_tag: tag or path of the discriminator model
- pretrained_classifier_model: tag or path of the style classifier model
- lambdas: loss weighting factors in the form "λ1|λ2|λ3|λ4|λ5" for cycle-consistency, generator, discriminator (fake), discriminator (real), and classifier-guided losses, respectively
- path_mono_A/path_mono_B: path to the training dataset for style A/B
- path_mono_A_eval/path_mono_B_eval: path to the validation dataset for style A/B (if references for validation are not available, as in the Yelp dataset)
- path_paral_A_eval/path_paral_B_eval: path to the validation dataset for style A/B (if references for validation are available, as in the GYAFC dataset)
- path_paral_eval_ref: path to the references for validation (if references available, as in the GYAFC dataset)
- learning_rate, epochs, batch_size: learning rate, number of epochs and batch size for model training

As an example, to train the CycleGAN architecture for formality transfer using the GYAFC dataset (*Family & Relationships* domain), you can use the following command:
```
CUDA_VISIBLE_DEVICES=0 python train.py --style_a=informal --style_b=formal --lang=en \
                       --path_mono_A=./data/GYAFC/family_relationships/train.0.txt --path_mono_B=./data/GYAFC/family_relationships/train.1.txt \
                       --path_paral_A_eval=./data/GYAFC/family_relationships/dev.0.txt --path_paral_B_eval=./data/GYAFC/family_relationships/dev.1.txt --path_paral_eval_ref=./data/GYAFC/family_relationships/references/dev/ --n_references=4 --shuffle \
                       --generator_model_tag=google-t5/t5-large --discriminator_model_tag=distilbert-base-cased --pretrained_classifier_model=./classifiers/GYAFC/family_relationships/bert-base-cased_5/ \
                       --lambdas="10|1|1|1|1" --epochs=30 --learning_rate=5e-5 --max_sequence_length=64 --batch_size=8  \
                       --save_base_folder=./ckpts/ --save_steps=1 --eval_strategy=epochs --eval_steps=1  --pin_memory --use_cuda_if_available
```

## Testing
Once trained, you can evaluate the performance on the test set of the trained models using the `test.py` script. It can be customized using several command line arguments such as:
- style_a/style_b: style A/B (i.e., informal/formal or negative/positive)
- generator_model_tag: tag or path of the generator model
- discriminator_model_tag: tag or path of the discriminator model
- from_pretrained: folder to use as base path to load the model checkpoint(s) to test
- pretrained_classifier_eval: tag or path of the oracle classifier model
- path_paral_A_test/path_paral_B_test: path to the test dataset for style A/B
- path_paral_test_ref: path to the references for test

As an example, to test the trained models for formality transfer using the GYAFC dataset (*Family & Relationships* domain), you can use the following command:
```
CUDA_VISIBLE_DEVICES=0 python test.py --style_a=informal --style_b=formal --lang=en \
                       --path_paral_A_test=./data/GYAFC/family_relationships/test.0.txt --path_paral_B_test=./data/GYAFC/family_relationships/test.1.txt --path_paral_test_ref=./data/GYAFC/family_relationships/references/test/ --n_references=4 \
                       --generator_model_tag=google-t5/t5-large --discriminator_model_tag=distilbert-base-cased \
                       --pretrained_classifier_eval=./classifiers/GYAFC/family_relationships/bert-base-cased_5/ \
                       --from_pretrained=./ckpts/ --max_sequence_length=64 --batch_size=16 --pin_memory --use_cuda_if_available 
```

## Model checkpoints
The checkpoints of the best-performing models for both Text Style Transfer tasks will be made available on Hugging Face 🤗 in the following weeks. We will include the corresponding links to the model checkpoints below.

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Authors
[Moreno La Quatra](https://mlaquatra.me/), [Giuseppe Gallipoli](https://scholar.google.com/citations?user=uMRKRW0AAAAJ&hl=it), [Luca Cagliero](https://scholar.google.it/citations?user=0uIAXl8AAAAJ&hl=it)

### Corresponding author
For any questions about the content of the paper or the implementation, you can contact me at: `giuseppe[DOT]gallipoli[AT]polito[DOT]it`.

## Citation
If you find this work useful, please cite our paper:

```bibtex
@article{LaQuatra24TST,
author = {La Quatra, Moreno and Gallipoli, Giuseppe and Cagliero, Luca},
title = {Self-supervised Text Style Transfer using Cycle-Consistent Adversarial Networks},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {2157-6904},
url = {https://doi.org/10.1145/3678179},
doi = {10.1145/3678179},
journal = {ACM Trans. Intell. Syst. Technol.},
month = {jul},
keywords = {Text Style Transfer, Sentiment transfer, Formality transfer, Cycle-consistent Generative Adversarial Networks, Transformers}
}
```

