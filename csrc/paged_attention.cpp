#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

// bfloat16 is the top 16 bits of a float32.
inline float to_float(const uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// Python's bfloat16 matmul sums in float32 and rounds each result, so rounding at the same
// points is what keeps this function matching the Python source of truth.
inline float round_bf16(const float value) { return static_cast<float>(c10::BFloat16(value)); }

void check_inputs(const torch::Tensor& query, const torch::Tensor& keys,
                  const torch::Tensor& values, const std::vector<int64_t>& block_table,
                  const int64_t length, const int64_t block_size, const int64_t groups) {
  TORCH_CHECK(query.dtype() == torch::kBFloat16 && keys.dtype() == torch::kBFloat16 &&
                  values.dtype() == torch::kBFloat16,
              "query, keys and values must be bfloat16");
  TORCH_CHECK(query.dim() == 2, "query must have shape (heads, head_dim)");
  TORCH_CHECK(keys.dim() == 4, "keys must have shape (kv_heads, blocks, block_size, head_dim)");
  TORCH_CHECK(keys.sizes().equals(values.sizes()), "keys and values must have the same shape");
  TORCH_CHECK(keys.is_contiguous() && values.is_contiguous(), "keys and values must be contiguous");
  TORCH_CHECK(block_size > 0 && block_size == keys.size(2), "block_size does not match keys");
  TORCH_CHECK(query.size(1) == keys.size(3), "query and keys disagree on head_dim");
  TORCH_CHECK(groups > 0 && query.size(0) == keys.size(0) * groups,
              "heads must equal kv_heads * groups");
  TORCH_CHECK(length > 0, "length must be positive");
  const int64_t blocks_needed = (length + block_size - 1) / block_size;
  TORCH_CHECK(static_cast<int64_t>(block_table.size()) >= blocks_needed,
              "block table is too short for length");
  for (int64_t index = 0; index < blocks_needed; ++index) {
    TORCH_CHECK(block_table[index] >= 0 && block_table[index] < keys.size(1),
                "block table entry out of range");
  }
}

// Walks the blocks in place, so K/V are never copied into contiguous memory.
void score_one_head(const uint16_t* query, const uint16_t* keys,
                    const std::vector<int64_t>& block_table, const int64_t length,
                    const int64_t block_size, const int64_t head_dim, const float scaling,
                    float* scores) {
  for (int64_t position = 0; position < length; ++position) {
    const int64_t block = block_table[position / block_size];
    const uint16_t* key = keys + (block * block_size + position % block_size) * head_dim;
    float dot = 0.0f;
    for (int64_t dim = 0; dim < head_dim; ++dim) {
      dot += to_float(query[dim]) * to_float(key[dim]);
    }
    scores[position] = round_bf16(round_bf16(dot) * scaling);
  }
}

void sum_values_one_head(const float* weights, const uint16_t* values,
                         const std::vector<int64_t>& block_table, const int64_t length,
                         const int64_t block_size, const int64_t head_dim, float* output) {
  std::fill(output, output + head_dim, 0.0f);
  for (int64_t position = 0; position < length; ++position) {
    const int64_t block = block_table[position / block_size];
    const uint16_t* value = values + (block * block_size + position % block_size) * head_dim;
    const float weight = round_bf16(weights[position]);
    for (int64_t dim = 0; dim < head_dim; ++dim) {
      output[dim] += weight * to_float(value[dim]);
    }
  }
}

torch::Tensor paged_attention(const torch::Tensor& query_in, const torch::Tensor& keys,
                              const torch::Tensor& values, const std::vector<int64_t>& block_table,
                              const int64_t length, const int64_t block_size, const int64_t groups,
                              const double scaling) {
  check_inputs(query_in, keys, values, block_table, length, block_size, groups);
  const torch::Tensor query = query_in.contiguous();
  const int64_t heads = query.size(0);
  const int64_t head_dim = query.size(1);
  const int64_t kv_stride = keys.size(1) * block_size * head_dim;
  const auto* query_data = reinterpret_cast<const uint16_t*>(query.data_ptr());
  const auto* key_data = reinterpret_cast<const uint16_t*>(keys.data_ptr());
  const auto* value_data = reinterpret_cast<const uint16_t*>(values.data_ptr());
  torch::Tensor scores = torch::empty({heads, length}, torch::kFloat32);
  torch::Tensor output = torch::empty({heads, head_dim}, torch::kFloat32);
  // Released so that a long attention call cannot stall the server's event loop.
  pybind11::gil_scoped_release release;
  for (int64_t head = 0; head < heads; ++head) {
    score_one_head(query_data + head * head_dim, key_data + (head / groups) * kv_stride,
                   block_table, length, block_size, head_dim, static_cast<float>(scaling),
                   scores.data_ptr<float>() + head * length);
  }
  // PyTorch's own softmax, because its vectorized exp rounds differently from std::exp.
  const torch::Tensor weights = torch::softmax(scores, -1);
  for (int64_t head = 0; head < heads; ++head) {
    sum_values_one_head(weights.data_ptr<float>() + head * length,
                        value_data + (head / groups) * kv_stride, block_table, length, block_size,
                        head_dim, output.data_ptr<float>() + head * head_dim);
  }
  return output.to(torch::kBFloat16);
}

}  // namespace

PYBIND11_MODULE(paged_attention_cpp, module) {
  module.def("paged_attention", &paged_attention);
}
