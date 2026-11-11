#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct sequence {
    llama_seq_id id;
    std::vector<llama_token> tokens;
};

using vector = std::vector<float>;
using embeddings = std::map<llama_seq_id, std::vector<vector>>;

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int32_t size = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    if (size <= 0) {
        throw std::runtime_error("tokenizer did not return a positive size");
    }
    std::vector<llama_token> tokens(size);
    if (llama_tokenize(vocab, text.data(), text.size(), tokens.data(), size, true, true) != size) {
        throw std::runtime_error("tokenization failed");
    }
    return tokens;
}

embeddings run(llama_context * ctx, const std::vector<sequence> & sequences, bool pooled, int offset = 0) {
    if (auto memory = llama_get_memory(ctx)) {
        llama_memory_clear(memory, false);
    }
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx), llama_batch_ext_free);
    for (const auto & seq : sequences) {
        for (int i = 0; i < int(seq.tokens.size()); ++i) {
            const llama_pos pos = i + offset;
            const int index = llama_batch_ext_add_token(batch.get(), seq.id, seq.tokens[i]);
            if (index < 0 || !llama_batch_ext_set_pos(batch.get(), index, &pos) ||
                !llama_batch_ext_set_output_embd(batch.get(), index, true)) {
                throw std::runtime_error("extended batch could not use the supplied sequence and position");
            }
        }
    }
    const auto result = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (result != 0) {
        throw std::runtime_error("decode failed with code " + std::to_string(result));
    }
    llama_synchronize(ctx);
    const int dim = llama_model_n_embd_out(llama_get_model(ctx));
    embeddings output;
    int batch_index = 0;
    for (const auto & seq : sequences) {
        const int count = pooled ? 1 : seq.tokens.size();
        for (int i = 0; i < count; ++i) {
            const float * values = pooled ? llama_get_embeddings_seq(ctx, seq.id) :
                llama_get_embeddings_ith(ctx, batch_index + i);
            if (!values || !std::all_of(values, values + dim, [](float v) { return std::isfinite(v); })) {
                throw std::runtime_error("missing or invalid embedding");
            }
            output[seq.id].emplace_back(values, values + dim);
        }
        batch_index += seq.tokens.size();
    }
    return output;
}

double delta(const std::vector<vector> & left, const std::vector<vector> & right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("embedding count mismatch");
    }
    double max_delta = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        for (size_t j = 0; j < left[i].size(); ++j) {
            max_delta = std::max(max_delta, std::abs(double(left[i][j]) - right[i][j]));
        }
    }
    return max_delta;
}

double cosine(const vector & left, const vector & right) {
    double dot = 0, left_norm = 0, right_norm = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        dot += double(left[i]) * right[i];
        left_norm += double(left[i]) * left[i];
        right_norm += double(right[i]) * right[i];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

double min_cosine(const std::vector<vector> & left, const std::vector<vector> & right) {
    double result = 1;
    for (size_t i = 0; i < left.size(); ++i) {
        result = std::min(result, cosine(left[i], right.at(i)));
    }
    return result;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        return 2;
    }
    try {
        ggml_backend_load_all_from_path(argv[1]);
        llama_log_set([](ggml_log_level level, const char * message, void *) {
            if (level >= GGML_LOG_LEVEL_WARN) std::fputs(message, stderr);
        }, nullptr);
        llama_backend_init();
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[3], model_params), llama_model_free);
        if (!model) {
            throw std::runtime_error("model load failed");
        }
        const auto * vocab = llama_model_get_vocab(model.get());
        const sequence a{1, tokenize(vocab, "白い猫")};
        const sequence b{3, tokenize(vocab, "黒い犬")};
        auto changed_b = b;
        const auto replacement = tokenize(vocab, "鳥が飛んだ");
        if (replacement.size() < 3) throw std::runtime_error("replacement too short for control");
        for (size_t i = 0; i < changed_b.tokens.size(); ++i) {
            if (i > 0 && i + 1 < changed_b.tokens.size()) {
                changed_b.tokens[i] = replacement.at(1 + (i - 1) % (replacement.size() - 2));
            }
        }
        if (changed_b.tokens == b.tokens) throw std::runtime_error("mutation did not change any tokens");
        const sequence concat_ab{1, tokenize(vocab, "白い猫 黒い犬")};
        const sequence concat_ba{1, tokenize(vocab, "黒い犬 白い猫")};
        auto params = llama_context_default_params();
        params.n_ctx = params.n_batch = params.n_ubatch = std::stoi(argv[5]);
        params.n_seq_max = 4;
        params.kv_unified = true;
        params.n_threads = params.n_threads_batch = 4;
        params.embeddings = true;
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        const auto model_pooling = std::string(argv[4]) == "last" ? LLAMA_POOLING_TYPE_LAST : LLAMA_POOLING_TYPE_MEAN;
        for (auto mode : {model_pooling, LLAMA_POOLING_TYPE_NONE}) {
            params.pooling_type = mode;
            const bool pooled = mode != LLAMA_POOLING_TYPE_NONE;
            std::unique_ptr<llama_context, decltype(&llama_free)> context(
                llama_init_from_model(model.get(), params), llama_free);
            if (!context) {
                throw std::runtime_error("context setup failed");
            }
            auto * ctx = context.get();
            const auto ab = run(ctx, {a, b}, pooled);
            const auto repeat = run(ctx, {a, b}, pooled);
            const auto ba = run(ctx, {b, a}, pooled);
            const auto altered = run(ctx, {a, changed_b}, pooled);
            const auto altered_swapped = run(ctx, {changed_b, a}, pooled);
            std::unique_ptr<llama_context, decltype(&llama_free)> fresh_context(
                llama_init_from_model(model.get(), params), llama_free);
            if (!fresh_context) throw std::runtime_error("fresh context setup failed");
            const auto ba_fresh = run(fresh_context.get(), {b, a}, pooled);
            const auto canonical_run = [&](std::vector<sequence> sequences) {
                std::sort(sequences.begin(), sequences.end(), [](const auto & left, const auto & right) {
                    return left.tokens == right.tokens ? left.id < right.id : left.tokens < right.tokens;
                });
                return run(ctx, sequences, pooled, 1);
            };
            const auto canonical_ab = canonical_run({a, b});
            const auto canonical_ba = canonical_run({b, a});
            const auto only_a = run(ctx, {a}, pooled);
            const auto only_b = run(ctx, {b}, pooled);
            const auto relabeled = run(ctx, {{3, a.tokens}, {1, b.tokens}}, pooled);
            const double order_delta = std::max(delta(ab.at(1), ba.at(1)), delta(ab.at(3), ba.at(3)));
            const double independent_delta = std::max(delta(ab.at(1), only_a.at(1)), delta(ab.at(3), only_b.at(3)));
            const double relabel_delta = std::max(delta(ab.at(1), relabeled.at(3)), delta(ab.at(3), relabeled.at(1)));
            const double order_cos = std::min(min_cosine(ab.at(1), ba.at(1)), min_cosine(ab.at(3), ba.at(3)));
            const double repeat_delta = std::max(delta(ab.at(1), repeat.at(1)), delta(ab.at(3), repeat.at(3)));
            const double fresh_delta = std::max(delta(ba.at(1), ba_fresh.at(1)), delta(ba.at(3), ba_fresh.at(3)));
            const double isolation_delta = std::max(delta(ab.at(1), altered.at(1)), delta(ba.at(1), altered_swapped.at(1)));
            const double b_delta = delta(ab.at(3), altered.at(3));
            const double canonical_delta = std::max(delta(canonical_ab.at(1), canonical_ba.at(1)),
                delta(canonical_ab.at(3), canonical_ba.at(3)));
            std::cout << "{\"model\":\"" << argv[2] << "\",\"pooling\":\"" << (pooled ? argv[4] : "none")
                      << "\",\"tokens\":[" << a.tokens.size() << "," << b.tokens.size()
                      << "],\"dimension\":" << ab.at(1).front().size()
                      << ",\"reorder_max_delta\":" << order_delta
                      << ",\"independent_max_delta\":" << independent_delta
                      << ",\"relabel_max_delta\":" << relabel_delta
                      << ",\"reorder_min_cosine\":" << order_cos
                      << ",\"repeat_max_delta\":" << repeat_delta
                      << ",\"fresh_context_max_delta\":" << fresh_delta
                      << ",\"cross_sequence_mutation_max_delta\":" << isolation_delta
                      << ",\"mutated_sequence_max_delta\":" << b_delta
                      << ",\"canonical_one_based_reorder_max_delta\":" << canonical_delta;
            if (pooled) {
                const auto x = run(ctx, {concat_ab}, true);
                const auto y = run(ctx, {concat_ba}, true);
                std::cout << ",\"concat_reorder_cosine\":" << cosine(x.at(1).front(), y.at(1).front());
            }
            if (std::string(argv[2]) == "minilm") {
                const auto shifted = run(ctx, {a, b}, pooled, 1);
                std::cout << ",\"one_based_shift_max_delta\":" << delta(ab.at(1), shifted.at(1));
            }
            std::cout << "}" << std::endl;
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
