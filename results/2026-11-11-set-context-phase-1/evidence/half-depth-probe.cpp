#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

struct Text {
    std::string name;
    Tokens tokens;
    int content_begin;
    int content_end;
};

Tokens tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    const int size = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, add_special, false);
    if (size <= 0) throw std::runtime_error("empty tokenization");
    Tokens tokens(size);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, add_special, false) != size) {
        throw std::runtime_error("tokenization failed");
    }
    return tokens;
}

Text encode(const llama_vocab * vocab, const std::string & text) {
    const auto plain = tokenize(vocab, text, false);
    auto full = tokenize(vocab, text, true);
    const auto content = std::search(full.begin(), full.end(), plain.begin(), plain.end());
    if (content == full.end()) throw std::runtime_error("content is not a contiguous span: " + text);
    const int begin = int(content - full.begin());
    return {text, std::move(full), begin, begin + int(plain.size())};
}

double cosine(const Vector & left, const Vector & right) {
    const auto dot = [](const Vector & a, const Vector & b) {
        return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
    };
    return dot(left, right) / std::sqrt(dot(left, left) * dot(right, right));
}

double max_delta(const Vector & left, const Vector & right) {
    double delta = 0;
    for (size_t i = 0; i < left.size(); ++i) delta = std::max(delta, std::abs(left[i] - right.at(i)));
    return delta;
}

Vector normalized(Vector value) {
    const double norm = std::sqrt(std::inner_product(value.begin(), value.end(), value.begin(), 0.0));
    if (!(norm > 0)) throw std::runtime_error("zero-norm set embedding");
    for (auto & coordinate : value) coordinate /= norm;
    return value;
}

struct Result {
    std::map<std::string, Vector> members;
    Vector token_mean;
    Vector set_embedding;
    json probe;
};

Result evaluate(llama_context * ctx, std::vector<Text> members, int blocked_layers, bool canonical = true) {
    if (canonical) {
        std::sort(members.begin(), members.end(), [](const auto & a, const auto & b) { return a.name < b.name; });
    }
    if (auto * memory = llama_get_memory(ctx)) llama_memory_clear(memory, false);
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
    std::vector<std::pair<llama_pos, size_t>> layout;
    for (size_t i = 0; i < members.size(); ++i) {
        for (llama_pos position = 0; position < int(members[i].tokens.size()); ++position) {
            layout.emplace_back(position, i);
        }
    }
    // One native sequence shares attention. Keep its positions nondecreasing
    // while each original member still uses its own zero-based positions.
    std::stable_sort(layout.begin(), layout.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    std::vector<std::vector<int>> rows(members.size());
    for (size_t i = 0; i < layout.size(); ++i) {
        const auto & [position, member] = layout[i];
        const int index = llama_batch_ext_add_token(batch.get(), 0, members[member].tokens[position]);
        if (index != int(i) || !llama_batch_ext_set_pos(batch.get(), index, &position) ||
                !llama_batch_ext_set_output_embd(batch.get(), index, true)) {
            throw std::runtime_error("cannot pack member");
        }
        rows[member].push_back(index);
    }
    const auto * model = llama_get_model(ctx);
    const int layers = llama_model_n_layer(model);
    const int heads = llama_model_n_head(model);
    const int dim = llama_model_n_embd_out(model);
    std::vector<std::vector<llama_attention_span>> own(members.size()), other(members.size());
    std::vector<llama_attention_rule> rules;
    if (blocked_layers > 0) {
        for (size_t i = 0; i < members.size(); ++i) {
            for (int j = 0; j < int(layout.size()); ++j) {
                auto & spans = layout[j].second == i ? own[i] : other[i];
                if (!spans.empty() && spans.back().end == j) spans.back().end = j + 1;
                else spans.push_back({j, j + 1});
            }
            rules.push_back({(uint64_t{1} << blocked_layers) - 1, (uint64_t{1} << heads) - 1,
                own[i].data(), own[i].size(), other[i].data(), other[i].size(), false});
        }
    }
    const int query = rows[0][members[0].content_end - 1];
    const int key = rows[1][members[1].content_begin];
    llama_attention_probe probes[] = {{layers / 2 - 1, 0, query, key, false, false, 0},
                                      {layers / 2, 0, query, key, false, false, 0}};
    const char * error = nullptr;
    const int status = llama_process_with_attention(ctx, batch.get(), rules.data(), rules.size(), probes, 2, &error);
    if (status != 0) {
        throw std::runtime_error(std::string("inference failed: ") + (error ? error : std::to_string(status)));
    }
    Result result{{}, Vector(dim, 0), Vector(dim, 0), json::array()};
    for (const auto & probe : probes) {
        if (!probe.observed) throw std::runtime_error("mask probe was not observed");
        result.probe.push_back({{"layer", probe.layer}, {"allowed", probe.allowed}, {"probability", probe.probability}});
    }
    int lexical_tokens = 0;
    for (size_t i = 0; i < members.size(); ++i) {
        const auto & member = members[i];
        Vector mean(dim, 0);
        for (int j = member.content_begin; j < member.content_end; ++j) {
            const float * row = llama_get_embeddings_ith(ctx, rows[i][j]);
            if (!row || !std::all_of(row, row + dim, [](float x) { return std::isfinite(x); })) {
                throw std::runtime_error("missing token vector");
            }
            ++lexical_tokens;
            for (int k = 0; k < dim; ++k) {
                mean[k] += row[k];
                result.token_mean[k] += row[k];
            }
        }
        for (auto & value : mean) value /= member.content_end - member.content_begin;
        result.members.emplace(member.name, std::move(mean));
    }
    for (auto & value : result.token_mean) value /= lexical_tokens;
    for (const auto & entry : result.members) {
        const auto unit = normalized(entry.second);
        for (int k = 0; k < dim; ++k) result.set_embedding[k] += unit[k];
    }
    result.set_embedding = normalized(std::move(result.set_embedding));
    return result;
}

json contexts(llama_context * ctx, const llama_vocab * vocab, const std::string & target,
              const std::vector<std::string> & a, const std::vector<std::string> & b,
              const std::string & ref_a, const std::string & ref_b, int layers) {
    const auto pack = [&](const std::vector<std::string> & words) {
        std::vector<Text> members{encode(vocab, target)};
        for (const auto & word : words) members.push_back(encode(vocab, word));
        return members;
    };
    const auto a_members = pack(a);
    const auto b_members = pack(b);
    const auto isolated_a = evaluate(ctx, a_members, layers);
    const auto isolated_b = evaluate(ctx, b_members, layers);
    const auto contextual_a = evaluate(ctx, a_members, layers / 2);
    const auto contextual_b = evaluate(ctx, b_members, layers / 2);
    const auto anchor_a = evaluate(ctx, pack({ref_a}), layers / 2).members.at(target);
    const auto anchor_b = evaluate(ctx, pack({ref_b}), layers / 2).members.at(target);

    auto reversed = a_members;
    std::reverse(reversed.begin(), reversed.end());
    const auto canonical_reverse = evaluate(ctx, reversed, layers / 2);
    const auto raw_reverse = evaluate(ctx, reversed, layers / 2, false);

    const auto & va = contextual_a.members.at(target);
    const auto & vb = contextual_b.members.at(target);
    const auto & ia = isolated_a.members.at(target);
    const auto & ib = isolated_b.members.at(target);
    return {
        {"target", target}, {"context_a", a}, {"context_b", b}, {"anchor_context_a", ref_a}, {"anchor_context_b", ref_b},
        {"isolated_context_cosine", cosine(ia, ib)}, {"isolated_context_max_delta", max_delta(ia, ib)},
        {"contextual_context_cosine", cosine(va, vb)}, {"contextual_context_max_delta", max_delta(va, vb)},
        {"isolated_anchor_cosines", {{cosine(ia, anchor_a), cosine(ia, anchor_b)},
                                     {cosine(ib, anchor_a), cosine(ib, anchor_b)}}},
        {"contextual_anchor_cosines", {{cosine(va, anchor_a), cosine(va, anchor_b)},
                                       {cosine(vb, anchor_a), cosine(vb, anchor_b)}}},
        {"canonical_reverse_max_delta", max_delta(va, canonical_reverse.members.at(target))},
        {"set_canonical_reverse_max_delta", max_delta(contextual_a.set_embedding, canonical_reverse.set_embedding)},
        {"raw_reverse_cosine", cosine(va, raw_reverse.members.at(target))},
        {"isolated_cross_member_probe", isolated_a.probe},
        {"contextual_cross_member_probe", contextual_a.probe},
    };
}

std::vector<std::pair<size_t, llama_token>> position_signature(const std::vector<Text> & members) {
    std::vector<std::pair<size_t, llama_token>> positions;
    for (const auto & member : members) {
        for (size_t i = 0; i < member.tokens.size(); ++i) positions.emplace_back(i, member.tokens[i]);
    }
    std::sort(positions.begin(), positions.end());
    return positions;
}

json pair_binding(llama_context * ctx, const llama_vocab * vocab, int layers) {
    const std::vector<Text> original{encode(vocab, "白い猫"), encode(vocab, "黒い犬")};
    const std::vector<Text> reversed{original[1], original[0]};
    const std::vector<Text> reassigned{encode(vocab, "白い犬"), encode(vocab, "黒い猫")};
    const auto a = evaluate(ctx, original, layers / 2);
    const auto b = evaluate(ctx, reassigned, layers / 2);
    const auto independent_a = evaluate(ctx, original, layers);
    const auto independent_b = evaluate(ctx, reassigned, layers);
    const auto shared_a = evaluate(ctx, original, 0);
    const auto shared_b = evaluate(ctx, reassigned, 0);
    return {
        {"original", {"白い猫", "黒い犬"}}, {"reassigned", {"白い犬", "黒い猫"}},
        {"same_token_position_multiset", position_signature(original) == position_signature(reassigned)},
        {"independent_member_pool_cosine", cosine(independent_a.set_embedding, independent_b.set_embedding)},
        {"contextual_member_pool_cosine", cosine(a.set_embedding, b.set_embedding)},
        {"contextual_token_pool_cosine", cosine(a.token_mean, b.token_mean)},
        {"fully_shared_token_pool_cosine", cosine(shared_a.token_mean, shared_b.token_mean)},
        {"canonical_reverse_max_delta", max_delta(a.set_embedding, evaluate(ctx, reversed, layers / 2).set_embedding)},
    };
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2;
    try {
        llama_log_set([](ggml_log_level level, const char * message, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(message, stderr);
        }, nullptr);
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], model_params), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        auto params = llama_context_default_params();
        params.n_ctx = params.n_batch = params.n_ubatch = 128;
        params.n_seq_max = 1;
        params.n_threads = params.n_threads_batch = 4;
        params.embeddings = true;
        params.pooling_type = LLAMA_POOLING_TYPE_NONE;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), params), llama_free);
        if (!ctx) throw std::runtime_error("context setup failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int layers = llama_model_n_layer(model.get());
        json rows = json::array();
        rows.push_back(contexts(ctx.get(), vocab, "ジャガー", {"ヒョウ", "チーター"}, {"BMW", "ベンツ"},
            "食肉目ネコ科の大型哺乳類", "英国の高級自動車メーカー", layers));
        rows.push_back(contexts(ctx.get(), vocab, "マウス", {"ハムスター", "ラット"}, {"キーボード", "トラックパッド"},
            "げっ歯類の小動物", "パソコンのポインティング機器", layers));
        rows.push_back(contexts(ctx.get(), vocab, "bank", {"river", "stream"}, {"money", "credit"},
            "sloping land at the water's edge", "an institution providing financial services", layers));
        std::cout << json{{"model", argv[1]}, {"layers", layers}, {"blocked_early_layers", layers / 2},
                          {"dimension", llama_model_n_embd_out(model.get())}, {"contexts", rows},
                          {"pair_binding", pair_binding(ctx.get(), vocab, layers)}}.dump() << '\n';
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
