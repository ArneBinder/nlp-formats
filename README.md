# nlp-formats
NLP formats for [huggingface/nlp](https://github.com/huggingface/nlp)

This library intents to separate NLP dataset formats from the actual datasets to make them reusable and signifcantly decrease dataset onboarding effort.

## Available Dataset Formats
* [Brat](https://brat.nlplab.org/): use `AbstractBrat`
* [Germeval2014](http://www.lrec-conf.org/proceedings/lrec2014/pdf/276_Paper.pdf): use `AbstractGermeval2014`

## Usage
See [sciarg.py](https://github.com/ArneBinder/nlp/blob/dataset_sciarg/datasets/sciarg/sciarg.py) for an example.
