#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "curand.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory>
#include <time.h>
#include <fstream>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
#define checkCurandErrors(val) check_curand( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(99);
	}
}

void check_curand(curandStatus_t result, char const *const func, const char *const file, int const line)
{
	if (result != CURAND_STATUS_SUCCESS)
	{
		std::cerr << "CURAND error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		cudaDeviceReset();
		exit(101);
	}
}

__global__ void mulKernel(float *c, const float *a, const float *b, int n)
{
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;
	float result = 0.0f;
	if (row < n && col < n) {
		for (int i = 0; i < n; i++) {
			result += a[row * n + i] * b[i * n + col];
		}
		c[row * n + col] = result;
	}
	
}

void mulCPU(float *c, const float *a, const float *b, int n)
{
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			c[n*j + i] = 0.0f;
	for (int i = 0; i < n; ++i)
		for (int j = 0; j < n; ++j)
			for (int k = 0; k < n; ++k)
				c[n*j + i] += a[k* n + i] * b[j * n + k];
}

void fill_matrix(float *a, int n)
{
	int size = n*n;
	curandGenerator_t gen;
	checkCurandErrors(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	checkCurandErrors(curandSetPseudoRandomGeneratorSeed(gen, clock()));
	checkCurandErrors(curandGenerateUniform(gen, a, size));

}

void print_matrix(float *c, int n)
{
	printf("\n");
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			printf("%.2f ", c[n*j + i]);
		}
		printf("\n");
	}
}

void fill_test(float *a)
{
	float t[3][3] = { {1,2,3},{4,5,6},{7,8,9}};
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			a[j*3 + i] = t[i][j];
}

int main()
{
	cudaDeviceProp prop;
	checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
	printf("Your CUDA-compatible device: %s\n", prop.name);

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	std::ofstream out("out.txt");

	for (int n = 200; n < 1600; n += 200)
	{
		int size = n * n * sizeof(float);
		float *a = (float*)malloc(size);
		float *b = (float*)malloc(size);
		float *c = (float*)malloc(size);

		//fill_test(a);
		//fill_test(b);
		fill_matrix(a, n);
		fill_matrix(b, n);
		fill_matrix(c, n);

		float *a_gpu, *b_gpu, *c_gpu;

		checkCudaErrors(cudaMalloc((void**)&a_gpu, size));
		checkCudaErrors(cudaMalloc((void**)&b_gpu, size));
		checkCudaErrors(cudaMalloc((void**)&c_gpu, size));

		checkCudaErrors(cudaMemcpy(a_gpu, a, size, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(b_gpu, b, size, cudaMemcpyHostToDevice));

		checkCudaErrors(cudaEventRecord(start));

		int tx = 16, ty = 16;
		dim3 blocks(n / tx + 1, n / ty + 1);
		dim3 threads(tx, ty);

		mulKernel << <blocks, threads >> > (c_gpu, a_gpu, b_gpu, n);
		checkCudaErrors(cudaEventRecord(stop));
		checkCudaErrors(cudaEventSynchronize(stop));
		cudaDeviceSynchronize();

		float milliseconds = 0;
		checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

		milliseconds /= 1000.0f;

		printf("\nGPU calculating time: %.5f", milliseconds);

		checkCudaErrors(cudaMemcpy(c, c_gpu, size, cudaMemcpyDeviceToHost));

		clock_t start1 = clock();
		mulCPU(c, a, b, n);

		float elapsed = difftime(clock(), start1) / CLOCKS_PER_SEC;

		printf("\nCPU calculating time: %.5f", elapsed);

		printf("\nSpeedup: %.2f\n", elapsed / milliseconds);

		out << n << " " << milliseconds << " " << elapsed << " " << (elapsed / milliseconds) << std::endl;

		free(a);
		free(b);
		free(c);

		checkCudaErrors(cudaFree(a_gpu));
		checkCudaErrors(cudaFree(b_gpu));
		checkCudaErrors(cudaFree(c_gpu));
	}
	out.close();
	return 0;
}