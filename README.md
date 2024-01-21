# Automatic Speech Recognition project

This repository contains ASR project done as a part of homework #1 for the DLA course at the CS Faculty of HSE. See [wandb report](https://wandb.ai/crimsonsparrow048/asr_project/reports/ASR-DLA-report--Vmlldzo2NTg2NzQw?accessToken=pd3dblext9ymignqvn2c53tni4rmlfivxct1y7812yper2ecnjou0iamx1nlx97j). 

## Installation guide

Clone this repository. Move to corresponding folder and install required packages:

```shell
git clone https://github.com/ir4kgl/asr_dla
cd asr_dla
pip install -r ./requirements.txt
```

## Checkpoint

To download the final checkpoint run 

```shell
python3 download_checkpoint.py
```

it will download final checkpoint in `checkpoints/final_run` folder.

## Run train

To train model simply run

```shell
python3 train.py --c config.json --r CHECKPOINT.pth
```

where `config.json` is configuration file with all data, model and trainer parameters and `CHECKPOINT.pth` is an optional argument to continue training starting from a given checkpoint. 

Configuration of my final experiment you can find in the file `configs/final_run_config.json`.


## Run evaluation

To evaluate a checkpoint run 

```shell
python3 eval.py --c config.json --r CHECKPOINT.pth
```

where `config.json` is configuration file with all data and model parameters.

Beam search with LM-model is used in my final evaluation run.  I used a `3-gram` model downladable by this code below:

```shell
wget 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
gunzip /kaggle/working/3-gram.pruned.1e-7.arpa.gz
```

However it does not work right away and before we start using this model we have to make it lower-case. So run the following code:

```shell
mv 3-gram.pruned.1e-7.arpa  data/upper_3-gram.pruned.1e-7.arpa
python lm_model_fix.py
```

It will move downloaded model to `data/` folder (that's exactly where test script expects to find it) and fix the case right away.


To run the evaluation with my final checkpoint you can use the configuration `configs/deepspeech_test_clean.json` to evaluate on test-clean split. After running the following command

```shell
python3 test.py --c configs/deepspeech_test_clean.json --r checkpoints/final_run/checkpoint.pth 
```

a file `output.json` with corresponding fields: `ground_truth`, `pred_text_argmax`, `pred_beam_search`, `wer_argm`, `cer_argm`, `wer_bs` and `cer_bs` will be created for every item in test data split:

*  `ground_truth` -- a target text  
 * `pred_text_argmax` -- predictions obtained with simple argmax
 * `pred_beam_search` -- predictions obtained with beam-search (with LM-model in final case)
 * `wer_argm`-- WER-metric calculated for argmax predictions
 * `cer_argm` --  CER-metric calculated for argmax predictions
 * `wer_bs` -- WER-metric calculated for beam-search predictions
 * `cer_bs` -- CER-metric calculated for beam-search predictions


Note that the last element in output file is `SUMMARY`, mean metrics calculated for all of the instances (so this value is used as a final score).

Also note that if you use `configs/deepspeech_test_clean.json` without any modifications, you have to put Librispeech dataset into `data/` folder.


I attach my final checkpoint's evaluation results in this GitHub repository (`final_output_asr.json`). Final WER score `wer_bs` my model gets is 0.145

I do not attach results for test-other split as there is nothing to catch there (but you can test with this data running code above with config `configs/deepspeech_test_other.json`)


