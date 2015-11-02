"""
Generate natural language sentences using a language model.

Usage:
  generate.py -i <file> -n <n>
  generate.py -h | --help

Options:
  -i <file>     Language model file.
  -n <n>        Number of sentences to generate.
  -h --help     Show this screen.

 """
import pickle
from docopt import docopt
from languagemodeling.ngram import NGramGenerator

if __name__ == '__main__':

    opts = docopt(__doc__)
    # train the model
    model_file = str(opts['-i'])
    model = pickle.load(open(model_file, "rb"))
    # train the model
    n = int(opts['-n'])
    generator = NGramGenerator(model)
    for i in range(n):
        print(generator.generate_sent())
