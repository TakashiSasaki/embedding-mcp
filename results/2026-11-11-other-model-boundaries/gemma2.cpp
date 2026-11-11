#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;
using Tokens = std::vector<llama_token>;
using Vector = std::vector<double>;

struct Member {
    std::string text;
    Tokens tokens;
    size_t content_begin;
    size_t content_end;
};

Tokens tokenize(const llama_vocab * vocab, const std::string & text, bool special) {
    const int size = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, special, false);
    if (size <= 0) throw std::runtime_error("Empty tokenization: " + text);
    Tokens tokens(size);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, special, false) != size) {
        throw std::runtime_error("Tokenization failed: " + text);
    }
    return tokens;
}

Member encode(const llama_vocab * vocab, const std::string & text) {
    const auto plain = tokenize(vocab, text, false);
    auto full = tokenize(vocab, text, true);
    const auto start = std::search(full.begin(), full.end(), plain.begin(), plain.end());
    if (start == full.end()) throw std::runtime_error("Content is not contiguous: " + text);
    const auto begin = size_t(start - full.begin());
    return {text, std::move(full), begin, begin + plain.size()};
}

double dot(const Vector & a, const Vector & b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double cosine(const Vector & a, const Vector & b) {
    return std::clamp(dot(a, b) / std::sqrt(dot(a, a) * dot(b, b)), -1.0, 1.0);
}

void normalize(Vector & vector) {
    const auto norm = std::sqrt(dot(vector, vector));
    if (!(norm > 0) || !std::isfinite(norm)) throw std::runtime_error("Invalid embedding norm");
    for (auto & value : vector) value /= norm;
}

double max_delta(const Vector & a, const Vector & b) {
    double maximum = 0;
    for (size_t i = 0; i < a.size(); ++i) maximum = std::max(maximum, std::abs(a[i] - b.at(i)));
    return maximum;
}

struct Embeddings {
    std::map<std::string, Vector> members;
    Vector token_mean;
    Vector set;
    json mask_checks = json::array();
};

Embeddings evaluate(llama_context * ctx, std::vector<Member> members, int blocked_layers,
                    bool canonical = true, bool inspect_mask = false) {
    if (canonical) {
        std::sort(members.begin(), members.end(), [](const auto & a, const auto & b) { return a.text < b.text; });
    }
    if (auto * memory = llama_get_memory(ctx)) llama_memory_clear(memory, false);
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx), llama_batch_ext_free);

    // The single native sequence uses nondecreasing positions. Each member
    // restarts at zero, so the batch is interleaved by position.
    std::vector<std::pair<llama_pos, size_t>> layout;
    for (size_t member = 0; member < members.size(); ++member) {
        for (llama_pos pos = 0; pos < int(members[member].tokens.size()); ++pos) layout.emplace_back(pos, member);
    }
    std::stable_sort(layout.begin(), layout.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    std::vector<std::vector<int>> rows(members.size());
    for (size_t i = 0; i < layout.size(); ++i) {
        const auto & [pos, member] = layout[i];
        const int index = llama_batch_ext_add_token(batch.get(), 0, members[member].tokens[pos]);
        if (index != int(i) || !llama_batch_ext_set_pos(batch.get(), index, &pos) ||
                !llama_batch_ext_set_output_embd(batch.get(), index, true)) {
            throw std::runtime_error("Cannot pack token at batch index " + std::to_string(i));
        }
        rows[member].push_back(index);
    }

    const auto * model = llama_get_model(ctx);
    const int layers = llama_model_n_layer(model);
    const int heads = llama_model_n_head(model);
    const int dimension = llama_model_n_embd_out(model);
    std::vector<std::vector<llama_attention_span>> own(members.size()), other(members.size());
    std::vector<llama_attention_rule> rules;
    if (blocked_layers > 0 && members.size() > 1) {
        for (size_t member = 0; member < members.size(); ++member) {
            for (int i = 0; i < int(layout.size()); ++i) {
                auto & spans = layout[i].second == member ? own[member] : other[member];
                if (!spans.empty() && spans.back().end == i) spans.back().end = i + 1;
                else spans.push_back({i, i + 1});
            }
            rules.push_back({(uint64_t{1} << blocked_layers) - 1, (uint64_t{1} << heads) - 1,
                own[member].data(), own[member].size(), other[member].data(), other[member].size(), false});
        }
    }

    std::vector<llama_attention_probe> probes;
    if (inspect_mask) {
        const int query = rows.at(0).at(members[0].content_begin);
        const int key = rows.at(1).at(members[1].content_begin);
        for (int layer = 0; layer < layers; layer += 3) probes.push_back({layer, 0, query, key, false, false, 0});
        for (int layer : {std::max(0, blocked_layers - 1), std::min(layers - 1, blocked_layers)}) {
            probes.push_back({layer, 0, key, query, false, false, 0});
            probes.push_back({layer, 0, query, query, false, false, 0});
        }
    }
    const char * error = nullptr;
    const auto status = llama_process_with_attention(ctx, batch.get(), rules.data(), rules.size(),
                                                    probes.data(), probes.size(), &error);
    if (status != 0) {
        throw std::runtime_error(std::string("Inference failed: ") + (error ? error : std::to_string(status)));
    }

    Embeddings out{{}, Vector(dimension, 0), Vector(dimension, 0)};
    for (const auto & probe : probes) {
        const bool expected = probe.query == probe.key || probe.layer >= blocked_layers;
        if (!probe.observed || probe.allowed != expected ||
                (!expected && probe.probability != 0) || !std::isfinite(probe.probability)) {
            throw std::runtime_error("Unexpected mask at layer " + std::to_string(probe.layer));
        }
        out.mask_checks.push_back({{"layer", probe.layer}, {"within_member", probe.query == probe.key},
                                   {"allowed", probe.allowed}, {"probability", probe.probability}});
    }
    size_t count = 0;
    for (size_t i = 0; i < members.size(); ++i) {
        const auto & member = members[i];
        Vector mean(dimension, 0);
        for (size_t j = member.content_begin; j < member.content_end; ++j) {
            const float * row = llama_get_embeddings_ith(ctx, rows[i][j]);
            if (!row || !std::all_of(row, row + dimension, [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("Missing token vector");
            }
            ++count;
            for (int d = 0; d < dimension; ++d) {
                mean[d] += row[d];
                out.token_mean[d] += row[d];
            }
        }
        for (auto & value : mean) value /= member.content_end - member.content_begin;
        normalize(mean);
        if (!out.members.emplace(member.text, std::move(mean)).second) throw std::runtime_error("Duplicate member");
    }
    for (auto & value : out.token_mean) value /= count;
    for (const auto & [text, vector] : out.members) {
        for (int d = 0; d < dimension; ++d) out.set[d] += vector[d];
    }
    normalize(out.set);
    return out;
}

std::vector<Member> encode_all(const llama_vocab * vocab, const json & texts) {
    std::vector<Member> members;
    for (const auto & text : texts) members.push_back(encode(vocab, text.get<std::string>()));
    return members;
}

std::vector<std::pair<size_t, llama_token>> signature(const std::vector<Member> & members) {
    std::vector<std::pair<size_t, llama_token>> pairs;
    for (const auto & member : members) {
        for (size_t pos = 0; pos < member.tokens.size(); ++pos) pairs.emplace_back(pos, member.tokens[pos]);
    }
    std::sort(pairs.begin(), pairs.end());
    return pairs;
}

int check_permutations(llama_context * ctx, std::vector<Member> members, int k, const Embeddings & expected) {
    const auto less = [](const Member & a, const Member & b) { return a.text < b.text; };
    std::sort(members.begin(), members.end(), less);
    int count = 1;
    while (std::next_permutation(members.begin(), members.end(), less)) {
        const auto actual = evaluate(ctx, members, k);
        if (actual.set != expected.set || actual.members != expected.members) {
            throw std::runtime_error("Canonical permutation changed an embedding at k=" + std::to_string(k));
        }
        ++count;
    }
    return count;
}

Vector definition_anchor(llama_context * ctx, const llama_vocab * vocab, const Member & target,
                         const std::string & definition) {
    auto entry = encode(vocab, target.text + " : " + definition);
    const size_t length = target.content_end - target.content_begin;
    if (entry.content_end - entry.content_begin < length ||
            !std::equal(target.tokens.begin() + target.content_begin, target.tokens.begin() + target.content_end,
                        entry.tokens.begin() + entry.content_begin)) {
        throw std::runtime_error("Definition changed target tokenization: " + target.text);
    }
    entry.content_end = entry.content_begin + length;
    return evaluate(ctx, {entry}, 0).members.at(entry.text);
}

struct SenseCase {
    std::string id, language, target;
    std::vector<Member> a, b;
    Vector anchor_a, anchor_b, legacy_a, legacy_b;
};

SenseCase prepare_sense(llama_context * ctx, const llama_vocab * vocab, const json & item) {
    SenseCase result{item.at("id"), item.at("language"), item.at("target"), {}, {}, {}, {}, {}, {}};
    const auto target = encode(vocab, result.target);
    const auto companions = [&](const char * side) {
        auto out = encode_all(vocab, item.at(side).at("companions"));
        out.insert(out.begin(), target);
        return out;
    };
    const auto definition_a = item.at("a").at("reference").get<std::string>();
    const auto definition_b = item.at("b").at("reference").get<std::string>();
    result.a = companions("a");
    result.b = companions("b");
    result.anchor_a = definition_anchor(ctx, vocab, target, definition_a);
    result.anchor_b = definition_anchor(ctx, vocab, target, definition_b);
    result.legacy_a = evaluate(ctx, {target, encode(vocab, definition_a)}, llama_model_n_layer(llama_get_model(ctx)) / 2).members.at(result.target);
    result.legacy_b = evaluate(ctx, {target, encode(vocab, definition_b)}, llama_model_n_layer(llama_get_model(ctx)) / 2).members.at(result.target);
    return result;
}

json anchor_scores(const Vector & a, const Vector & b, const Vector & ref_a, const Vector & ref_b) {
    const double aa = cosine(a, ref_a), ab = cosine(a, ref_b);
    const double ba = cosine(b, ref_a), bb = cosine(b, ref_b);
    return {{"cosines", {{aa, ab}, {ba, bb}}}, {"margins", {aa - ab, bb - ba}}};
}

json sense_scores(llama_context * ctx, const SenseCase & item, int k) {
    const auto a = evaluate(ctx, item.a, k), b = evaluate(ctx, item.b, k);
    const auto & target_a = a.members.at(item.target);
    const auto & target_b = b.members.at(item.target);
    const int orders = check_permutations(ctx, item.a, k, a) + check_permutations(ctx, item.b, k, b);
    auto reversed = item.a;
    std::reverse(reversed.begin(), reversed.end());
    const auto raw = evaluate(ctx, reversed, k, false);
    return {{"id", item.id}, {"language", item.language}, {"target", item.target},
            {"context_distance", 1 - cosine(target_a, target_b)},
            {"definition", anchor_scores(target_a, target_b, item.anchor_a, item.anchor_b)},
            {"legacy", anchor_scores(target_a, target_b, item.legacy_a, item.legacy_b)},
            {"orders_checked", orders}, {"raw_reverse_target_distance", 1 - cosine(target_a, raw.members.at(item.target))},
            {"raw_reverse_set_distance", 1 - cosine(a.set, raw.set)},
            {"raw_reverse_target_max_delta", max_delta(target_a, raw.members.at(item.target))}};
}

json binding_scores(llama_context * ctx, const llama_vocab * vocab, const json & item, int k, bool inspect_mask) {
    const auto original = encode_all(vocab, item.at("original"));
    const auto swapped = encode_all(vocab, item.at("reassigned"));
    const auto paraphrase = encode_all(vocab, item.at("paraphrase"));
    if (signature(original) != signature(swapped)) throw std::runtime_error("Token positions differ: " + item.at("id").dump());
    const auto a = evaluate(ctx, original, k, true, inspect_mask);
    const auto b = evaluate(ctx, swapped, k), p = evaluate(ctx, paraphrase, k);
    const int orders = check_permutations(ctx, original, k, a) + check_permutations(ctx, swapped, k, b) +
                       check_permutations(ctx, paraphrase, k, p);
    auto reversed = original;
    std::reverse(reversed.begin(), reversed.end());
    const double swapped_cosine = cosine(a.set, b.set);
    const double paraphrase_cosine = cosine(a.set, p.set);
    return {{"id", item.at("id")}, {"language", item.at("language")}, {"relation", item.at("relation")},
            {"swap_distance", 1 - swapped_cosine}, {"paraphrase_distance", 1 - paraphrase_cosine},
            {"ranking_margin", paraphrase_cosine - swapped_cosine},
            {"token_mean_swap_distance", 1 - cosine(a.token_mean, b.token_mean)}, {"orders_checked", orders},
            {"raw_reverse_set_distance", 1 - cosine(a.set, evaluate(ctx, reversed, k, false).set)},
            {"mask_checks", a.mask_checks}};
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "usage: set-context MINI_LM_GGUF FIXTURE_JSON\n";
        return 2;
    }
    try {
        llama_log_set([](ggml_log_level level, const char * message, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(message, stderr);
        }, nullptr);
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], model_params), llama_model_free);
        if (!model) {
            throw std::runtime_error("Expected the 12-layer, 12-head MiniLM model");
        }
        auto params = llama_context_default_params();
        params.n_ctx = params.n_batch = params.n_ubatch = 128;
        params.n_seq_max = 1;
        params.n_threads = params.n_threads_batch = 4;
        params.embeddings = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), params), llama_free);
        if (!ctx) throw std::runtime_error("Context setup failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        std::ifstream input(argv[2]);
        const auto fixture = json::parse(input);
        std::vector<SenseCase> senses;
        for (const auto & item : fixture.at("sense")) senses.push_back(prepare_sense(ctx.get(), vocab, item));
        for (int k = 0; k <= llama_model_n_layer(model.get()); ++k) {
            json sense = json::array(), binding = json::array();
            for (const auto & item : senses) sense.push_back(sense_scores(ctx.get(), item, k));
            for (const auto & item : fixture.at("binding")) {
                binding.push_back(binding_scores(ctx.get(), vocab, item, k, binding.empty()));
            }
            std::cout << json{{"k", k}, {"sense", sense}, {"binding", binding}}.dump() << std::endl;
            std::cerr << "Finished k=" << k << '\n';
        }
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
