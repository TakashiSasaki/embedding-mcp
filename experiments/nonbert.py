import ast
from collections import namedtuple
import gzip
import logging
from pathlib import Path
import re
import resource
import statistics
import time
from typing import IO, List

from common import ASSETS
from text import check_vectors, cosine, inputs, ranking, translation_top1


def load_class(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    assert {node.name for node in classes} == set(names)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), namespace)


def make_laser(threads):
    import numpy as np
    import sentencepiece as spm
    from sacremoses import MosesDetokenizer, MosesPunctNormalizer
    import torch
    from unicategories import categories

    torch.set_num_threads(threads)
    assets = ASSETS / 'ref-laser'
    namespace = {
        'np': np, 'torch': torch, 'nn': torch.nn,
        'spm': spm, 'MosesPunctNormalizer': MosesPunctNormalizer, 'MosesDetokenizer': MosesDetokenizer,
        'Path': Path, 'IO': IO, 'List': List, 'gzip': gzip,
        'logger': logging.getLogger('laser2'),
        'SPACE_NORMALIZER': re.compile(r'\s+'),
        'NON_PRINT_CHARS': set(categories['C'].characters()),
        'Batch': namedtuple('Batch', 'srcs tokens lengths'),
    }
    load_class(assets / 'laser_tokenizer.py', ('LaserTokenizer',), namespace)
    load_class(assets / 'models.py', ('LaserLstmEncoder', 'SentenceEncoder'), namespace)
    tokenizer = namespace['LaserTokenizer'](spm_model=assets / 'laser2.spm')
    encoder = namespace['SentenceEncoder'](model_path=assets / 'laser2.pt', cpu=True)
    assert isinstance(encoder.encoder, namespace['LaserLstmEncoder'])

    def embed(text):
        return encoder.encode_sentences([tokenizer.tokenize(text)], normalize_embeddings=True)[0].tolist()

    return embed


def make_potion():
    from model2vec import StaticModel

    model = StaticModel.from_pretrained(ASSETS / 'ref-potion', force_download=False)

    def embed(text):
        return model.encode([text], normalize=True, max_length=128, use_multiprocessing=False)[0].tolist()

    return embed


def benchmark(model, threads):
    embed, dimensions = (make_laser(threads), 1024) if model == 'laser2' else (make_potion(), 256)
    embed('Warm-up')
    elapsed = []
    vectors = {}
    for key, text in inputs(model).items():
        start = time.perf_counter()
        vectors[key] = embed(text)
        elapsed.append(1000 * (time.perf_counter() - start))
    check_vectors(vectors, dimensions)
    order_cosine = cosine(embed('The dog chased the cat.'), embed('The cat chased the dog.'))
    if model == 'potion':
        assert order_cosine > 0.99999
    else:
        assert order_cosine < 0.9999
    return {
        'ranking': ranking(vectors), 'translation_top1': translation_top1(vectors), 'median_ms': round(statistics.median(elapsed), 4),
        'peak_process_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        'order_reversal_cosine': round(order_cosine, 7), 'vectors': vectors,
    }
