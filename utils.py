import os
import pprint
import json
import cPickle
import numpy as np
pp = pprint.PrettyPrinter()

def save_pkl(path, obj):
  with open(path, 'w') as f:
    cPickle.dump(obj, f)
    print(" [*] save %s" % path)

def load_pkl(path):
  with open(path) as f:
    obj = cPickle.load(f)
    print(" [*] load %s" % path)
    return obj

def save_npy(path, obj):
  np.save(path, obj)
  print(" [*] save %s" % path)

def load_npy(path):
  obj = np.load(path)
  print(" [*] load %s" % path)
  return obj

def load_json(file):
  try:
    with open(file, 'r') as datafile:
      data = json.load(datafile)
  except Exception as e:
    raise e
  return data

def dump_json(data, file):
  try:
    with open(file, 'w') as datafile:
      json.dump(data, datafile)
  except Exception as e:
    raise e

def write_file(data, file):
  try:
    with open(file, 'w') as datafile:
      for line in data:
        datafile.write(' '.join([str(x) for x in line]) + '\n')
  except Exception as e:
    raise e

def revdict(d):
  """
  Reverse a dictionary mapping.
  When two keys map to the same value, only one of them will be kept in the
  result (which one is kept is arbitrary).
  """
  return dict((v, k) for (k, v) in d.iteritems())

def generate_corpus(corpus, vocab):
  rvocab = revdict(vocab)
  docs = {}
  for key, doc in corpus.iteritems():
    tokens = []
    for idx, val in doc.items():
      v = rvocab[int(idx)]
      tokens.extend([v for i in range(val)])
    docs[key] = tokens
  return docs

if __name__ == '__main__':
  import sys
  corpus = load_json(sys.argv[1])
  test_corpus = load_json(sys.argv[2])
  out_dir = sys.argv[3]
  docs, vocab_dict = corpus['docs'].items(), corpus['vocab']
  test_docs = test_corpus['docs']

  np.random.seed(0)
  np.random.shuffle(docs)
  n_val = 1000
  train_docs = dict(docs[:-n_val])
  val_docs = dict(docs[-n_val:])

  train_docs = generate_corpus(train_docs, vocab_dict)
  val_docs = generate_corpus(val_docs, vocab_dict)
  test_docs = generate_corpus(test_docs, vocab_dict)

  dump_json(train_docs, os.path.join(out_dir, 'train.json'))
  dump_json(val_docs, os.path.join(out_dir, 'valid.json'))
  dump_json(test_docs, os.path.join(out_dir, 'test.json'))
  dump_json(vocab_dict, os.path.join(out_dir, 'vocab.json'))

