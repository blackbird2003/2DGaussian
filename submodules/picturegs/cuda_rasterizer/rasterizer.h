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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:


		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P,
			const float* background,
			const int width, int height,
			const float* means2D,
			const float* colors,
			const float* opacities,
			const float* scales,
			const float* rots,
			const float* negative,
			const bool prefiltered,
			float* out_color,
			bool antialiasing,
			int* radii = nullptr,
			bool debug = false);

		static void backward(
			const int P,  int R,
			const float* background,
			const int width, int height,
			const float* colors,
			const float* opacities,
			const float* scalse,
			const float* rots,
			const float* negative,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			float* dL_dmean2D,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dconic,
			float* dL_dscale,
			float* dL_drot,
			float* dL_dnegative,
			bool antialiasing,
			bool debug);
	};
};

#endif
