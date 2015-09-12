#!/bin/bash
nosetests languagemodeling/tests/test_ngram.py
echo 'training uni-grams'
python languagemodeling/scripts/train.py -n 1 -o uni-gram.model
echo 'generating two sentences'
python languagemodeling/scripts/generate.py -n 2 -i uni-gram.model

echo 'training bi-grams'
python languagemodeling/scripts/train.py -n 2 -o bi-gram.model
echo 'generating two sentences'
python languagemodeling/scripts/generate.py -n 2 -i bi-gram.model

echo 'training tri-grams'
python languagemodeling/scripts/train.py -n 3 -o tri-gram.model
echo 'generating two sentences'
python languagemodeling/scripts/generate.py -n 2 -i tri-gram.model

echo 'training 4-grams'
python languagemodeling/scripts/train.py -n 4 -o four-gram.model
echo 'generating two sentences'
python languagemodeling/scripts/generate.py -n 2 -i four-gram.model

nosetests languagemodeling/tests/test_ngram_generator.py
nosetests languagemodeling/tests/test_addone_ngram.py


