import gzip
import os
import re
import tarfile
import sys

import six.moves import urllib

import tensorflow as tf

# Special vocab. symbols
_PAD = b"_PAD"
_GO  = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID  = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions for tokenization
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE   = re.compile(br"\d")

# URLs for WMT data
_WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
_WMT_ENFR_DEV_URL   = "http://www.statmt.org/wmt15/dev-v2.tgz"

def download(directory, filename, url):
    if not os.path.exists(directory):
        print("Creating dir. {}".format(directory))
        os.mkdir(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r >> Downloading %s %.1f%%' %(filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.requst.urlretrieve(_WMT_ENFR_TRAIN_URL, filepath, _progress)
        print()
        print("Successfully downloading", filename, os.stat(filepath).st_size, "bytes.")
    return filepath


def gunzip_file(gz_path, new_path, remove=True):
    print("Unpacking %s to %s" % (gz_path, new_path))
    with gzip.open(gz_path, "rb") as gz_file:
        with open(new_path, "wb") as new_file:
            for line in gz_file:
                new_file.write(line)
    
    if remove:
        os.remove(gz_path)


def get_train_set(directory):
    train_path = os.path.join(directory, "giga-fren.release2.fixed")
    if not (gfile.Exists(train_path + ".fr") and gfile.Exists(train_path + ".en")):
        corpus_file = download(directory, "training-giga-fren.tar",
                               _WMT_ENFR_TRAIN_URL)
        print("Extracting tar file %s" % corpus_file)
        with tarfile.open(corpus_file, "r") as corpus_tar:
            corpus_tar.extractall(directory)
        gunzip_file(train_path + ".fr.gz", train_path + ".fr")
        gunzip_file(train_path + ".en.gz", train_path + ".en")
    return train_path

def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(_WORD_SPLIT.split(space_separated_fragment))
    return [w for w in words if w]

def create_vocab(vocab_path, data_path, max_vocab_size,
                 tokenizer=None, nomalize_digits=True):
    if not gfile.Exists(vocab_path):
        print("Creating vocabulary %s from data %s" % (vocab_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                line = tf.compat.as_bytes(line)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = _DEGIT_RE.sub(b"0", w) if normalize_digis else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocab_size:
                vocab_list = vocab_list[:max_vocab+_size]
            with gfiel.GFile(vocab_path, mode="wb") as vocab_f:
                for w in vocab_list:
                    vocab_f.write(w + b"\n")

def init_vocab(vocab_path):
    if gfile.Exists(vocab_path):
        rev_vocab = []
        with gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
        vocab = dict([ (x,y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("vocab file %s not found.", vocab_path)

def sentence_to_token_ids(sentence, vocab,
                          tokenizer=None, normalize_digits=True):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocab.get(w, UNK_ID) for w in words]
    return [vocaburary.get(_DEGIT_RE.sub(b"0", w), UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocab_path,
                      tokenizer=None, normalize_digits=True):
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocab(vocab_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                for line in data_file:
                    token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                                                         tokenizer,
                                                                         normalize_digis)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_data(data_dir, en_vocab_size, fr_vocab_size, tokenizer=None):
    train_path = get_wmt_enfr_train_set(data_dir)
    dev_path   = get_wmt_enfr_dev_set(data_dir)

    from_train_path = train_path + ".en"
    to_train_path   = train_path + ".fr"
    from_dev_path   = dev_path   + ".en"
    to_dev_path     = dev_path   + ".fr"

    return prepare_data(data_dir, from_train_path, to_train_path,
                        from_dev_path, to_dev_path,
                        en_vocab_size, fr_vocab_size, tokenizer)

def prepare_data(data_dir, from_train_path, to_train_path,
                 from_dev_path, to_dev_path,
                 from_vocab_size, to_vocab_size, tokenizer=None):
    to_vocab_path   = os.path.join(data_dir, "vocab%d.to" % to_vocab_size)
    from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocab_size)
    create_vocab(to_vocab_path, to_train_path, to_vocab_size, tokenizer)
    create_vocab(from_vocab_path, from_train_path, from_vocab_size, tokenizer)

    to_train_ids_path   = to_train_path + (".ids%d" % to_vocab_size)
    from_train_ids_path = from_train_path + (".ids%d" % from_vocab_size)
    data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_size, tokenizer)
    data_to_token_ids(from_train_path, form_train_ids_path, from_vocab_size, tokenizer)
                        
    
    return (from_train_ids_path, to_train_ids_path,
            from_dev_ids_path, to_dev_ids_path,
            from_vocab_path, to_vocab_path)
                                        
                
