#include "kernel.h"
#include "workspace.cuh"
#include "config.h"

#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/gather.h>
#include <math.h>

#define BLOCK_SIZE 256

// Device helpers
// Grid (x, y) -> 1D value
__device__ int getGridHash(int gridPosX, int gridPosY) {
    gridPosX = max(0, min(gridPosX, COLUMNS_COUNT - 1));
    gridPosY = max(0, min(gridPosY, ROWS_COUNT - 1));
    return gridPosY * COLUMNS_COUNT + gridPosX;
}

// Get a grid cell of chosen particle (x,y) -> (gridCol, gridRow)
// P.s. assume x,y are from [-1,1] interval 
__device__ int2 getGridPosition(float x, float y) {
    int2 gridPos;
    gridPos.x = (int)((x + 1.0F) * 0.5F * COLUMNS_COUNT);
    gridPos.y = (int)((y + 1.0F) * 0.5F * ROWS_COUNT);
    return gridPos;
}

// Clip a color to ensure is is inside [0, 255] interval
__device__ unsigned char clipColor(float val) {
    return (unsigned char)(fminf(fmaxf(val, 0.0F), 255.0F));
}

// Kernel functions
// Find an index of grid cell, where every particle is situated + fill helper sort arrays 
__global__ void calculateHash(int* d_gridParticleHash, int* d_gridParticleIndex, float* d_posX, float* d_posY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PARTICLES_COUNT) 
        return;

    float px = d_posX[idx];
    float py = d_posY[idx];

    int2 gridPos = getGridPosition(px, py);
    int hash = getGridHash(gridPos.x, gridPos.y);

    d_gridParticleHash[idx] = hash;
    d_gridParticleIndex[idx] = idx;
}

// CellStart/CellEnd arrays reset 
__global__ void resetCellStartEnd(int* d_cellStart, int* d_cellEnd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ROWS_COUNT * COLUMNS_COUNT) 
        return;

    d_cellStart[idx] = -1;
    d_cellEnd[idx] = -1;
}

// Assume d_gridParticleHash is already sorted according to grid cell hashes. 
// I.e. | Hash | ParticleIndex |
//      |   0  |      5        |
//      |   0  |      7        |
//      |   1  |      2        |
//      |   1  |      3        |
//      |   1  |      8        |
//      |   2  |      1        |
//            ...
// This function finds start and end indices of particles in each cell and fills
// d_cellStart and d_cellEnd arrays accordingly. If a cell is empty, both values are -1.
// Thanks to this, while calculating the logic of hash0, we can iterate through much less 
// interval of particles (i.e. for hash0 we iterate only through particles with indices [5,7])
__global__ void findCellBounds(int* d_gridParticleHash, int* d_cellStart, int* d_cellEnd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PARTICLES_COUNT) 
        return;

    int myHash = d_gridParticleHash[idx];
    int prevHash = (idx == 0) ? -1 : d_gridParticleHash[idx - 1];

    // If hashes are different, we found the start of a new cell, so we set the start index
    // of this cell to the current particle index. We also set the end index of the previous
    // cell to the current particle index (exclusive).
    if (myHash != prevHash) 
    {
        d_cellStart[myHash] = idx;
        if (idx > 0 && prevHash != -1) 
        {
            d_cellEnd[prevHash] = idx;
        }
    }

    // Last element case  
    if (idx == PARTICLES_COUNT - 1) {
        d_cellEnd[myHash] = idx + 1;
    }
}

// Kernel function which calculates both field map and graphics output
__global__ void calculateFieldAndGraphics(uchar4* outputSurface, float2* d_fieldMap, float* d_sortedPosX,
    float* d_sortedPosY, float* d_sortedCharge, int* d_cellStart, int* d_cellEnd)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= WINDOW_WIDTH || y >= WINDOW_HEIGHT) 
        return;

    // Get normalized world coordinates in [-1, 1] range
    float u = (x / (float)WINDOW_WIDTH) * 2.0F - 1.0F;
    float v = (y / (float)WINDOW_HEIGHT) * 2.0F - 1.0F;

    int2 myGridPos = getGridPosition(u, v);
    float2 E = make_float2(0.0F, 0.0F);
    float potentialSum = 0.0F;

    // Check neighboring cells (3x3)
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {

            int nx = myGridPos.x + dx;
            int ny = myGridPos.y + dy;

            if (nx >= 0 && nx < COLUMNS_COUNT && ny >= 0 && ny < ROWS_COUNT) 
            {
                int hash = getGridHash(nx, ny);

                int startIndex = d_cellStart[hash];

                if (startIndex != -1) 
                {
                    int endIndex = d_cellEnd[hash];

                    for (int k = startIndex; k < endIndex; k++) {

                        float px = d_sortedPosX[k];
                        float py = d_sortedPosY[k];
                        float charge = d_sortedCharge[k];

                        float dX = u - px;
                        float dY = v - py;
                        float distSq = dX * dX + dY * dY + 0.001F;

                        if (sqrtf(distSq) < 0.04F)
                        {
                            float forceMag = charge / distSq;
                            E.x += forceMag * (dX / sqrtf(distSq));
                            E.y += forceMag * (dY / sqrtf(distSq));
                            potentialSum += charge / sqrtf(distSq);
                        }
                    }
                }
            }
        }
    }

    int pixelIndex = y * WINDOW_WIDTH + x;

    float maxField = 500.0F;
    E.x = fmaxf(-maxField, fminf(maxField, E.x));
    E.y = fmaxf(-maxField, fminf(maxField, E.y));
    d_fieldMap[pixelIndex] = E;

    /*
    uchar4 color = make_uchar4(0, 0, 0, 255);
    float intensity = fabs(potentialSum) * 30.0F;
    if (potentialSum > 0) 
    {
        color.x = clipColor(intensity);
    }
    else 
    {
        color.z = clipColor(intensity);
    }
    */

    uchar4 color = make_uchar4(0, 0, 0, 255);
    float maxVisualPotential = 100.0F;
    float intensity = fminf((fabs(potentialSum) / maxVisualPotential) * 255.0F, 255.0F);
    if (potentialSum > 0)
        color.x = clipColor(intensity);
    if (potentialSum < 0)
        color.z = clipColor(intensity);

    outputSurface[pixelIndex] = color;
}

// This function updates particle physics based on the precomputed field map
__global__ void updatePhycicsFromModel(float* d_posX, float* d_posY, float* d_velX, float* d_velY, float* d_charge, float2* d_fieldMap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PARTICLES_COUNT) 
        return;

    float px = d_posX[idx];
    float py = d_posY[idx];
    float vx = d_velX[idx];
    float vy = d_velY[idx];
    float charge = d_charge[idx];

    int mapX = (int)((px + 1.0F) / 2.0F * WINDOW_WIDTH);
    int mapY = (int)((py + 1.0F) / 2.0F * WINDOW_HEIGHT);

    mapX = max(0, min(mapX, WINDOW_WIDTH - 1));
    mapY = max(0, min(mapY, WINDOW_HEIGHT - 1));

    float2 E = d_fieldMap[mapY * WINDOW_WIDTH + mapX];

    // F = qE = ma -> a = qE/m
    float mass = (charge == ELECTRON_CHARGE) ? ELECTRON_MASS : PROTON_MASS;
    float ax = E.x * charge / mass;
    float ay = E.y * charge / mass;

    vx += ax * DT;
    vy += ay * DT;

    px += vx * DT;
    py += vy * DT;

    if (px > 1.0F) 
    { 
        px = 1.0F; 
        vx *= -0.8F; 
    }
    if (px < -1.0F) 
    { 
        px = -1.0F; 
        vx *= -0.8F; 
    }
    if (py > 1.0F) 
    { 
        py = 1.0F; 
        vy *= -0.8F; 
    }
    if (py < -1.0F) 
    { 
        py = -1.0F; 
        vy *= -0.8F; 
    }

    d_posX[idx] = px;
    d_posY[idx] = py;
    d_velX[idx] = vx;
    d_velY[idx] = vy;
}

struct ParticleVertex {
    float x, y;
    float r, g, b;
};

__global__ void packParticlesKernel(ParticleVertex* output, float* d_posX, float* d_posY, float* d_charge) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= PARTICLES_COUNT) return;

    ParticleVertex v;
    v.x = d_posX[idx];
    v.y = d_posY[idx];

    float q = d_charge[idx];
    if (q > 0.0f) 
    { 
        v.r = 0.9f; 
        v.g = 0.9f; 
        v.b = 0.9f; 
    } 
    else 
    { 
        v.r = 0.3f; 
        v.g = 0.3f; 
        v.b = 0.3f; 
    } 

    output[idx] = v;
}

void launchSimulation(workspace* ws, float time)
{
    int N = PARTICLES_COUNT;
    int blocksParticles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocksCells = (ROWS_COUNT * COLUMNS_COUNT + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Calculate Hashes
    calculateHash << <blocksParticles, BLOCK_SIZE >> > (ws->d_gridParticleHash, ws->d_gridParticleIndex, ws->d_posX, ws->d_posY);

    // 2. Sort Particles by Hash using Thrust
    thrust::device_ptr<int> t_hash(ws->d_gridParticleHash);
    thrust::device_ptr<int> t_index(ws->d_gridParticleIndex);
    thrust::sort_by_key(t_hash, t_hash + N, t_index);

    // 3. Reorder data
    thrust::device_ptr<float> t_posX(ws->d_posX);
    thrust::device_ptr<float> t_posY(ws->d_posY);
    thrust::device_ptr<float> t_charge(ws->d_charge);
    thrust::device_ptr<float> t_sortedPosX(ws->d_sortedPosX);
    thrust::device_ptr<float> t_sortedPosY(ws->d_sortedPosY);
    thrust::device_ptr<float> t_sortedCharge(ws->d_sortedCharge);

    thrust::gather(t_index, t_index + N, t_posX, t_sortedPosX);
    thrust::gather(t_index, t_index + N, t_posY, t_sortedPosY);
    thrust::gather(t_index, t_index + N, t_charge, t_sortedCharge);

    // 4. Reset Cell Start/End
    resetCellStartEnd << <blocksCells, BLOCK_SIZE >> > (ws->d_cellStart, ws->d_cellEnd);
    findCellBounds << <blocksParticles, BLOCK_SIZE >> > (ws->d_gridParticleHash, ws->d_cellStart, ws->d_cellEnd);

    // 5. Calculate Field Map and Graphics Output
    uchar4* d_out;
    size_t numBytes;
    cudaGraphicsMapResources(1, &ws->cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &numBytes, ws->cudaResource);

    dim3 block(16, 16);
    dim3 grid((WINDOW_WIDTH + block.x - 1) / block.x, (WINDOW_HEIGHT + block.y - 1) / block.y);
    calculateFieldAndGraphics << <grid, block >> > (d_out, ws->d_fieldMap, ws->d_sortedPosX, ws->d_sortedPosY, ws->d_sortedCharge,
         ws->d_cellStart, ws->d_cellEnd);
    cudaGraphicsUnmapResources(1, &ws->cudaResource, 0);

    // TODO: add shared memory optimization here

    // 6. Update Particle Physics from Field Map
    updatePhycicsFromModel << <blocksParticles, BLOCK_SIZE >> > (ws->d_posX, ws->d_posY, ws->d_velX, ws->d_velY, ws->d_charge, ws->d_fieldMap);

    // 7. Update Particle Visualization (VBO)
    ParticleVertex* d_particleGeo;
    cudaGraphicsMapResources(1, &ws->cudaParticleResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_particleGeo, &numBytes, ws->cudaParticleResource);
    packParticlesKernel << <blocksParticles, BLOCK_SIZE >> > (d_particleGeo, ws->d_posX, ws->d_posY, ws->d_charge);
    cudaGraphicsUnmapResources(1, &ws->cudaParticleResource, 0);

    // cudaGraphicsUnmapResources(1, &ws->cudaResource, 0);
    cudaDeviceSynchronize();
}
