/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "spatial.h"
#include "simple_knn.h"
#include <cuda_runtime.h>

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  cudaSetDevice(points.device().index());
  const int P = points.size(0);

  torch::Tensor means = torch::full({P}, 0.0, points.options().dtype(torch::kFloat32));  
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}