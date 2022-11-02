#include <ATen/Functions.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <string.h>
#include <omp.h>
#include <vector>
#include "lstm_postop.hpp"
#include "linear.hpp"
#include "lstm_int8.hpp"

namespace intel_mlperf {
std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm_int8(
  const at::Tensor& x,
  const at::Tensor& hx,
  const at::Tensor& cx,
  const std::vector<std::vector<at::Tensor>> all_weights,
  const std::vector<at::Scalar>& o_scale,
  const std::vector<at::Scalar>& in_scale,
  const std::vector<at::Scalar>& out_scale,
  const bool skip_quant_y){

  at::Tensor hy_layer, cy_layer;
  std::vector<at::Tensor> hy_list, cy_list;
  auto num_layers = all_weights.size();

  auto x_p = at::split(x,1);
  for (int64_t layer = 0; layer < num_layers; layer++) {
    auto weights_layer = all_weights[layer];
    auto skip_quant = (layer==(num_layers-1)) & skip_quant_y;
    std::tie(x_p, hy_layer, cy_layer) = lstm_rnnt_cell(
        x_p, hx[layer], cx[layer],
        weights_layer[0], weights_layer[1],
        weights_layer[2], weights_layer[3],
        o_scale[layer], in_scale[layer], out_scale[layer],
        skip_quant);
    hy_list.emplace_back(hy_layer);
    cy_list.emplace_back(cy_layer);
  }  
  auto hy = at::stack(hy_list, 0);
  auto cy = at::stack(cy_list, 0);
  auto xy = at::stack(x_p,0);
  return {xy, hy, cy};
}

std::tuple<std::vector<at::Tensor>, at::Tensor, at::Tensor> lstm_rnnt_cell(
  const std::vector<at::Tensor>& x,
  const at::Tensor& hx,
  const at::Tensor& cx,
  const at::Tensor& w_ih,
  const at::Tensor& w_hh,
  const at::Tensor& b_ih,
  const at::Tensor& b_hh,
  const c10::optional<at::Scalar>& o_scale,
  const c10::optional<at::Scalar>& in_scale,
  const c10::optional<at::Scalar>& out_scale,
  const bool skip_quant_y){
  auto size_x = x.size();
  std::vector<at::Tensor> gates_list(size_x);
  for(int i=0;i<size_x;i++){
    auto gates = linear(torch::squeeze(x[i],0), w_ih, b_ih, o_scale->toFloat(), 0);
    gates_list[i] = gates;
  }
  
  std::vector<at::Tensor> yt_list(size_x);
  auto hxx = hx;
  auto cxx = cx;
  for(int i=0;i<size_x;i++){
    gates_list[i] += linear(hxx, w_hh, b_hh, o_scale->toFloat(), 0);

    auto gates_chunk = torch::chunk(gates_list[i],4,1);
    auto y_p = lstm_postop(gates_chunk[0], gates_chunk[1], gates_chunk[2], gates_chunk[3], cxx, in_scale->toFloat(), out_scale->toFloat(), skip_quant_y);

    hxx = y_p[2];
    cxx = y_p[3];

    if(skip_quant_y)
      yt_list[i] = y_p[0];
    else
      yt_list[i] = y_p[1];
  }

  return {yt_list, hxx, cxx};
}
}