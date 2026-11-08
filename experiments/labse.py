import math
import statistics

from common import ASSETS, server
from text import MODELS, cosine, inputs, ranking


def project(vectors, dense):
    import torch

    return torch.nn.functional.normalize(
        torch.nn.functional.linear(torch.tensor(vectors), dense['linear.weight'], dense['linear.bias']).tanh()
    ).tolist()


def diagnose(stock, reference, out, threads):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    model_path, _, _ = MODELS['labse']
    texts = inputs('labse')
    original = reference['vectors']
    stock_vectors = stock['vectors']
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / 'ref-labse', local_files_only=True)
    token_ids = tokenizer(list(texts.values()))['input_ids']
    dense = load_file(ASSETS / 'ref-labse/2_Dense/model.safetensors')
    with server(model_path, out / 'labse-diagnosis.log', threads=threads, context=128, pooling='cls') as client:
        def raw(content):
            vector = client.post('/embedding', {'content': content, 'embd_normalize': -1})[0]['embedding']
            if isinstance(vector[0], list):
                assert len(vector) == 1
                vector = vector[0]
            assert len(vector) == 768
            return vector

        native_raw = [raw(text) for text in texts.values()]
        reference_ids_raw = [raw([ids]) for ids in token_ids]

    normalized = torch.nn.functional.normalize(torch.tensor(native_raw)).tolist()
    assert min(cosine(vec, stock_vectors[key]) for key, vec in zip(texts, normalized, strict=True)) > 0.99999
    variants = {'stock': list(stock_vectors.values()),
                'HF-tokenizer-only': torch.nn.functional.normalize(torch.tensor(reference_ids_raw)).tolist(),
                'official-Dense-Tanh-only': project(native_raw, dense),
                'HF-tokenizer-plus-official-Dense-Tanh': project(reference_ids_raw, dense)}
    summary = {'first_raw_cls_l2': math.sqrt(cosine(native_raw[0], native_raw[0])), 'variants': {}}
    for variant, vecs in variants.items():
        vectors = dict(zip(texts, vecs, strict=True))
        cosines = [cosine(vec, original[key]) for key, vec in vectors.items()]
        summary['variants'][variant] = {'min_cosine': round(min(cosines), 6),
                            'median_cosine': round(statistics.median(cosines), 6), 'ranking': ranking(vectors)}
    return summary
