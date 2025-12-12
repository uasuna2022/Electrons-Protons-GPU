#pragma once

#include <iostream>
#include <stdlib.h>
#include <vector>
#include "cuda_runtime.h"
#include "glad/glad.h"
#include "GLFW/glfw3.h"
#include "config.h"
#include <cuda_gl_interop.h>


using namespace std;

class workspace
{
private:
	void allocateMemoryGPU();
	void freeMemoryGPU();
	void checkErrorCUDA(cudaError_t err, char* msg);
public:
	// Host data
	vector<float> h_posX;
	vector<float> h_posY;
	vector<float> h_velX;
	vector<float> h_velY;
	vector<float> h_charge;

	// Device data
	float* d_posX;
	float* d_posY;
	float* d_velX;
	float* d_velY;
	float* d_charge;

	// Grid data
	int* d_gridParticleHash;
	int* d_gridParticleIndex;
	int* d_cellStart;
	int* d_cellEnd;

	// Screen data
	unsigned int texID;
	unsigned int pboID;
	struct cudaGraphicsResource* cudaResource;

	workspace();
	~workspace();
	void Initialize();
};
