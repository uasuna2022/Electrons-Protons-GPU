#include "workspace.cuh"
#include <iostream>
#include <random>

using namespace std;

workspace::workspace(): d_posX(nullptr), d_posY(nullptr), d_velX(nullptr), d_velY(nullptr),
	d_charge(nullptr), d_gridParticleHash(nullptr), d_gridParticleIndex(nullptr),
	d_cellEnd(nullptr), d_cellStart(nullptr), texID(0), pboID(0), cudaResource(nullptr),
    d_fieldMap(nullptr) { }

workspace::~workspace()
{
	freeMemoryGPU();
    if (cudaResource)
        cudaGraphicsUnregisterResource(cudaResource);
    if (pboID)
        glDeleteBuffers(1, &pboID);
    if (texID)
        glDeleteTextures(1, &texID);
}

void workspace::allocateMemoryGPU()
{
    size_t sizeFloat = PARTICLES_COUNT * sizeof(float);
    size_t sizeInt = PARTICLES_COUNT * sizeof(int);
    size_t sizeGrid = ROWS_COUNT * COLUMNS_COUNT * sizeof(int);
    size_t sizeField = WINDOW_HEIGHT * WINDOW_WIDTH * sizeof(float2);

    cudaError_t err = cudaSuccess;
    cudaMalloc((void**)&d_posX, sizeFloat);
    checkErrorCUDA(err, "Malloc d_posX");
    err = cudaSuccess;

    cudaMalloc((void**)&d_posY, sizeFloat);
    checkErrorCUDA(err, "Malloc d_posY");
    err = cudaSuccess;

    cudaMalloc((void**)&d_velX, sizeFloat);
    checkErrorCUDA(err, "Malloc d_velX");
    err = cudaSuccess;

    cudaMalloc((void**)&d_velY, sizeFloat);
    checkErrorCUDA(err, "Malloc d_velY");
    err = cudaSuccess;

    cudaMalloc((void**)&d_charge, sizeFloat);
    checkErrorCUDA(err, "Malloc d_charge");
    err = cudaSuccess;

    cudaMalloc((void**)&d_gridParticleHash, sizeInt);
    checkErrorCUDA(err, "Malloc d_gridParticleHash");
    err = cudaSuccess;

    cudaMalloc((void**)&d_gridParticleIndex, sizeInt);
    checkErrorCUDA(err, "Malloc d_gridParticleIndex");
    err = cudaSuccess;

    cudaMalloc((void**)&d_cellStart, sizeGrid);
    checkErrorCUDA(err, "Malloc d_cellStart");
    err = cudaSuccess;

    cudaMalloc((void**)&d_cellEnd, sizeGrid);
    checkErrorCUDA(err, "Malloc d_cellEnd");
    err = cudaSuccess;

    cudaMalloc((void**)&d_fieldMap, sizeField);
    checkErrorCUDA(err, "Malloc d_fieldMap");
    err = cudaSuccess;
}

void workspace::freeMemoryGPU()
{
    if (d_posX) 
        cudaFree(d_posX);
    if (d_posY) 
        cudaFree(d_posY);
    if (d_velX) 
        cudaFree(d_velX);
    if (d_velY) 
        cudaFree(d_velY);
    if (d_charge) 
        cudaFree(d_charge);

    if (d_gridParticleHash) 
        cudaFree(d_gridParticleHash);
    if (d_gridParticleIndex) 
        cudaFree(d_gridParticleIndex);
    if (d_cellStart) 
        cudaFree(d_cellStart);
    if (d_cellEnd) 
        cudaFree(d_cellEnd);

    if (d_fieldMap)
        cudaFree(d_fieldMap);
}

void workspace::Initialize()
{
    allocateMemoryGPU();
    h_posX.resize(PARTICLES_COUNT);
    h_posY.resize(PARTICLES_COUNT);
    h_velX.resize(PARTICLES_COUNT);
    h_velY.resize(PARTICLES_COUNT);
    h_charge.resize(PARTICLES_COUNT);

    mt19937 random_generator(time(NULL));

    uniform_real_distribution<float> position(-1.0F, 1.0F);
    uniform_real_distribution<float> velocities(-0.01F, 0.01F);
    uniform_int_distribution<int> charges(0, 1);

    for (int i = 0; i < PARTICLES_COUNT; i++)
    {
        h_posX[i] = position(random_generator);
        h_posY[i] = position(random_generator);
        h_velX[i] = velocities(random_generator);
        h_velY[i] = velocities(random_generator);
        h_charge[i] = (charges(random_generator) == 0) ? ELECTRON_CHARGE : PROTON_CHARGE;
    }

    size_t size = PARTICLES_COUNT * sizeof(float);
    
    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(d_posX, h_posX.data(), size, cudaMemcpyHostToDevice);
    checkErrorCUDA(err, "h_posX -> d_posX memcpy error\0");
    err = cudaSuccess;

    err = cudaMemcpy(d_posY, h_posY.data(), size, cudaMemcpyHostToDevice);
    checkErrorCUDA(err, "h_posY -> d_posY memcpy error\0");
    err = cudaSuccess;

    err = cudaMemcpy(d_velX, h_velX.data(), size, cudaMemcpyHostToDevice);
    checkErrorCUDA(err, "h_velX -> d_velX memcpy error\0");
    err = cudaSuccess;

    err = cudaMemcpy(d_velY, h_velY.data(), size, cudaMemcpyHostToDevice);
    checkErrorCUDA(err, "h_velY -> d_velY memcpy error\0");
    err = cudaSuccess;

    err = cudaMemcpy(d_charge, h_charge.data(), size, cudaMemcpyHostToDevice);
    checkErrorCUDA(err, "h_charge -> d_charge memcpy error\0");
    err = cudaSuccess;

    // Texture create
    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_2D, texID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Texture memory allocate 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WINDOW_WIDTH, WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);


    // Generate pixel buffer object
    glGenBuffers(1, &pboID);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pboID);

    // Memory reserving
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WINDOW_WIDTH * WINDOW_HEIGHT * 4 * sizeof(unsigned char), NULL, GL_DYNAMIC_COPY);

    err = cudaGraphicsGLRegisterBuffer(&cudaResource, pboID, cudaGraphicsMapFlagsWriteDiscard);
    checkErrorCUDA(err, "PBO registration error\0");
    if (err == cudaSuccess)
        cout << "Initialization completed correctly!" << endl;

    err = cudaMemset(d_fieldMap, 0, WINDOW_WIDTH * WINDOW_HEIGHT * sizeof(float2));
    checkErrorCUDA(err, "d_fieldMap memset error\0");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void workspace::checkErrorCUDA(cudaError_t err, char* msg)
{
    if (err != cudaSuccess)
    {
        cerr << "FATAL ERROR: " << msg << endl;
        cerr << "CUDA Error: " << cudaGetErrorString(err) << endl;
    }
}