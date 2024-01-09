# BERT-pytorch-for-learning

## This code is written in accordance with "https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/bert.html"

## Training BERT: 
python main.py

## Fine tuning BERT:
### (i.e. Natural Language Inference. We can use the features extracted by BERT to feed into the MLP, and through training and fine-tuning, determine whether the latter sentence can be inferred from the previous sentence. )
### This is generally divided into three types: entailment, contradiction, neutral.
python fine_tuning.py

## Test BERT:
### You can try looking at BERT's <class> output to see if the text pair used for the test is contextual.
python test_bert.py
