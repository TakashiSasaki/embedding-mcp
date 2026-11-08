import itertools
import json
import math
import statistics
import time

from common import ASSETS, FIXTURE, peak_rss, server

TOPICS = json.loads(FIXTURE.read_text())
TOPIC_IDS = [row["id"] for row in TOPICS]
if len(TOPIC_IDS) < 2 or len(TOPIC_IDS) != len(set(TOPIC_IDS)):
    raise ValueError(f"Expected at least two distinct topic IDs in {FIXTURE}")
MODELS = {
    'e5': (ASSETS / 'multilingual-e5-small-f16.gguf', 'mean', 384),
    'minilm': (ASSETS / 'paraphrase-multilingual-MiniLM-L12-118M-v2-F16.gguf', 'mean', 384),
    'labse': (ASSETS / 'labse.Q8_0.gguf', 'cls', 768),
    'gemma2': (ASSETS / 'embeddinggemma-2-Q8_0.gguf', 'mean', 768),
    'qwen3': (ASSETS / 'Qwen3-Embedding-0.6B-Q8_0.gguf', 'last', 1024),
}
LANG = ('ja', 'en')


def inputs(model):
    prefixes = {
        'e5': ('query: ', 'passage: '),
        'gemma2': ('task: search result | query: ', 'title: none | text: '),
        'qwen3': ('Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:', ''),
    }.get(model, ('', ''))
    return {
        f'{kind}:{language}:{row["id"]}': prefix + row[field][language]
        for kind, field, prefix in zip(('q', 'd'), ('query', 'document'), prefixes, strict=True)
        for language in LANG
        for row in TOPICS
    }


def cosine(left, right):
    return sum(x * y for x, y in zip(left, right, strict=True))


def ranking(vectors):
    result = {}
    for query_lang, doc_lang in itertools.product(LANG, repeat=2):
        ranks = []
        misses = []
        for topic in TOPIC_IDS:
            query = vectors[f'q:{query_lang}:{topic}']
            ranked = sorted(
                ((cosine(query, vectors[f'd:{doc_lang}:{doc}']), doc) for doc in TOPIC_IDS),
                reverse=True,
            )
            rank = 1 + [doc for _, doc in ranked].index(topic)
            ranks.append(rank)
            if rank > 1:
                misses.append([topic, ranked[0][1], rank])
        result[f'{query_lang}->{doc_lang}'] = {
            'top1': ranks.count(1),
            'mrr': round(statistics.mean(1 / rank for rank in ranks), 4),
            'misses': misses,
        }
    return result


def translation_top1(vectors):
    return {
        f'{left}->{right}': sum(
            max(TOPIC_IDS, key=lambda doc: cosine(vectors[f'd:{left}:{topic}'], vectors[f'd:{right}:{doc}'])) == topic
            for topic in TOPIC_IDS
        )
        for left, right in (('ja', 'en'), ('en', 'ja'))
    }


def check_vectors(vectors, dimensions):
    for name, vector in vectors.items():
        assert len(vector) == dimensions and all(map(math.isfinite, vector)), name
        assert math.isclose(cosine(vector, vector), 1, abs_tol=2e-5), name
    first, *_, last = vectors.values()
    assert cosine(first, last) < 0.999


def run_gguf(model, out, threads):
    path, pooling, dimensions = MODELS[model]
    with server(path, out / f'{model}.log', threads=threads, context=128, pooling=pooling) as client:
        def embed(text):
            return client.post('/v1/embeddings', {'input': [text]})['data'][0]['embedding']

        embed('Warm-up')
        vectors = {}
        latency_ms = []
        texts = inputs(model)
        for key, text in texts.items():
            start = time.perf_counter()
            vectors[key] = embed(text)
            latency_ms.append(1000 * (time.perf_counter() - start))
        tokens = {key: client.post('/tokenize', {'content': text, 'add_special': True})['tokens']
                  for key, text in texts.items()}
        memory_mib = peak_rss(client.pid)

    check_vectors(vectors, dimensions)
    return {'ranking': ranking(vectors), 'translation_top1': translation_top1(vectors),
            'median_ms': round(statistics.median(latency_ms), 1),
            'max_tokens': max(map(len, tokens.values())), 'peak_rss_mib': memory_mib,
            'vectors': vectors, 'tokens': tokens}


def run_reference(model, stock, threads):
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(threads)
    ref = ASSETS / f'ref-{model}'
    tokenizer = AutoTokenizer.from_pretrained(ref, local_files_only=True)
    encoder = AutoModel.from_pretrained(ref, local_files_only=True, attn_implementation='eager').eval()
    texts = inputs(model)
    encoded = tokenizer(list(texts.values()), return_tensors='pt', padding=True, truncation=True, max_length=128)
    valid = encoded['attention_mask']
    with torch.inference_mode():
        hidden = encoder(**encoded).last_hidden_state
        if model == 'labse':
            dense = load_file(ref / '2_Dense/model.safetensors')
            pooled = torch.nn.functional.linear(hidden[:, 0], dense['linear.weight'], dense['linear.bias']).tanh()
        else:
            pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True)
        embeddings = torch.nn.functional.normalize(pooled).tolist()
    vectors = dict(zip(texts, embeddings, strict=True))
    tokens = {key: row[mask.bool()].tolist() for key, row, mask in zip(texts, encoded['input_ids'], valid, strict=True)}
    check_vectors(vectors, MODELS[model][2])
    parity = {key: cosine(vector, stock['vectors'][key]) for key, vector in vectors.items()}
    token_differences = [key for key in texts if tokens[key] != stock['tokens'][key]]
    result = {'architecture': type(encoder).__name__, 'ranking': ranking(vectors), 'vectors': vectors,
              'token_mismatch_count': len(token_differences),
              'token_mismatch_examples': [[key, tokens[key], stock['tokens'][key]] for key in token_differences[:2]],
              'cosine_min': round(min(parity.values()), 6),
              'cosine_median': round(statistics.median(parity.values()), 6)}
    if model == 'minilm':
        result['mask_experiment'] = test_mask(encoder, tokenizer)
    return result


def test_mask(encoder, tokenizer):
    import torch

    encoded = tokenizer('自転車のタイヤがすぐしぼむ。', return_tensors='pt')
    length = encoded['input_ids'].shape[-1]
    query_pos, key_pos = 1, 2
    native = torch.zeros((1, 1, length, length), dtype=encoder.dtype)
    blocked = native.clone()
    blocked[0, 0, query_pos, key_pos] = -torch.inf
    with torch.inference_mode():
        baseline = encoder(**encoded, output_attentions=True)
        explicit = encoder(**dict(encoded, attention_mask=native), output_attentions=True)
        modified = encoder(**dict(encoded, attention_mask=blocked), output_attentions=True)
        assert torch.allclose(baseline.last_hidden_state, explicit.last_hidden_state, atol=1e-6, rtol=1e-6)

        def weights(output):
            return torch.stack([layer[0, :, query_pos, key_pos] for layer in output.attentions])

        native_weights, blocked_weights = weights(baseline), weights(modified)
        assert native_weights.max() > 0 and blocked_weights.count_nonzero() == 0
        pooled = [torch.nn.functional.normalize(out.last_hidden_state.mean(1)) for out in (baseline, modified)]
        sim = (pooled[0] * pooled[1]).sum().item()
        assert math.isfinite(sim) and sim < 0.999999
    return {'tokens': tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist()),
            'query_position': query_pos, 'key_position': key_pos, 'layers': len(baseline.attentions),
            'native_edge_weight_mean': native_weights.mean().item(),
            'blocked_edge_weight_max': blocked_weights.max().item(), 'pooled_cosine': sim}
