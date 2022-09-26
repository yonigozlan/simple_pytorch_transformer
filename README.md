# Pytorch implementation of the "Transformer"

This work is a simple implementation of the model described in the seminal paper [Attention is all you need](https://arxiv.org/abs/1706.03762)

The model is implemented in Pytorch and is tested on a translation task from English to German and from German to English.


This implementation uses [poetry](https://python-poetry.org/) as package and dependency manager.

### Usage
The model can be trained and tested using the command line.
#### Training the transformer
To train the transformer on a translation task, use this command:
```
python -m translation train --language en_de --epochs 8
```
#### Testing the transformer
The transformer can be tested in two different ways.
We can test the transformer on examples from the validation dataset:
```
python -m translation examples --language en_de --nb 10
```
Or we can test the transformer on any sentence in the source language:
```
python -m translation infer "The quick brown fox jumps over the lazy dog." --language en_de
```

*This work is inspired by [Harvard NLP annotated transformer](http://nlp.seas.harvard.edu/annotated-transformer/)*
