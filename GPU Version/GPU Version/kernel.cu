#include "kernel.h"
#include "workspace.cuh" // Tutaj musi być pełna definicja klasy
#include "config.h"      // Tu są stałe (GRID_COLUMNS, WINDOW_WIDTH itp.)

#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <math.h>

// Rozmiar bloku wątków (standard optymalizacyjny)
#define BLOCK_SIZE 256

// =================================================================================
// SEKCJA 1: FUNKCJE POMOCNICZE (DEVICE)
// =================================================================================

// Zamienia współrzędne (x,y) na 1D Hash komórki
__device__ int getGridHash(int gridPosX, int gridPosY) {
    // Clamp (zabezpieczenie przed wyjściem poza siatkę)
    gridPosX = max(0, min(gridPosX, COLUMNS_COUNT - 1));
    gridPosY = max(0, min(gridPosY, ROWS_COUNT - 1));
    return gridPosY * COLUMNS_COUNT + gridPosX;
}

// Oblicza, w której komórce siatki znajduje się cząstka (na podstawie pozycji -1 do 1)
__device__ int2 getGridPos(float x, float y) {
    int2 gridPos;
    // Mapowanie [-1, 1] -> [0, GRID_COLUMNS]
    gridPos.x = (int)((x + 1.0f) * 0.5f * COLUMNS_COUNT);
    gridPos.y = (int)((y + 1.0f) * 0.5f * ROWS_COUNT);
    return gridPos;
}

// Pomocnicza do kolorów
__device__ unsigned char clipColor(float val) {
    return (unsigned char)(fminf(fmaxf(val, 0.0f), 255.0f));
}

// =================================================================================
// SEKCJA 2: KERNELS (OBLICZENIA RÓWNOLEGŁE)
// =================================================================================

// KROK 1: Obliczanie hashy (do której komórki należy każda cząstka?)
__global__ void calcHash(
    int* d_gridParticleHash,
    int* d_gridParticleIndex,
    float* d_posX,
    float* d_posY,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // 1. Pobierz pozycję (SoA)
    float px = d_posX[idx];
    float py = d_posY[idx];

    // 2. Oblicz komórkę
    int2 gridPos = getGridPos(px, py);
    int hash = getGridHash(gridPos.x, gridPos.y);

    // 3. Zapisz wynik
    d_gridParticleHash[idx] = hash; // Klucz sortowania
    d_gridParticleIndex[idx] = idx; // Wartość (oryginalny indeks cząstki)
}

// KROK 2: Resetowanie granic komórek (przygotowanie tablicy)
__global__ void resetCellStartEnd(int* d_cellStart, int* d_cellEnd, int numCells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCells) {
        d_cellStart[idx] = -1; // -1 oznacza "brak cząstek w tej komórce"
        d_cellEnd[idx] = -1;
    }
}

// KROK 3: Znajdowanie granic komórek w posortowanej tablicy
// (Działa w czasie O(N) zamiast O(N^2))
__global__ void findCellBounds(
    int* d_gridParticleHash,
    int* d_cellStart,
    int* d_cellEnd,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Pobieramy hash mój i poprzednika
    int myHash = d_gridParticleHash[idx];
    int prevHash = (idx == 0) ? -1 : d_gridParticleHash[idx - 1];

    // Jeśli hashe są różne, to znaczy że zmieniła się komórka -> zapisz granice
    if (myHash != prevHash) {
        d_cellStart[myHash] = idx;
        if (idx > 0 && prevHash != -1) {
            d_cellEnd[prevHash] = idx;
        }
    }

    // Obsługa ostatniego elementu
    if (idx == numParticles - 1) {
        d_cellEnd[myHash] = idx + 1;
    }
}

// KROK 4: Aktualizacja Fizyki (Ruch + Odbijanie od ścian)
__global__ void updatePhysics(
    float* d_posX, float* d_posY,
    float* d_velX, float* d_velY,
    int numParticles,
    float dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    // Odczyt (SoA)
    float px = d_posX[idx];
    float py = d_posY[idx];
    float vx = d_velX[idx];
    float vy = d_velY[idx];

    // Ruch
    px += vx; // Zakładamy, że prędkość jest już przeskalowana przez DT przy inicjalizacji lub tutaj
    py += vy;

    // Odbicia od ścian [-1, 1]
    if (px > 1.0f) { px = 1.0f; vx *= -1.0f; }
    if (px < -1.0f) { px = -1.0f; vx *= -1.0f; }
    if (py > 1.0f) { py = 1.0f; vy *= -1.0f; }
    if (py < -1.0f) { py = -1.0f; vy *= -1.0f; }

    // Zapis (SoA)
    d_posX[idx] = px;
    d_posY[idx] = py;
    d_velX[idx] = vx;
    d_velY[idx] = vy;
}

// KROK 5: Wizualizacja (Rysowanie Pola do PBO)
__global__ void visualizeField(
    uchar4* outputSurface, // Wyjście graficzne
    int width, int height,
    // Dane cząstek (SoA)
    float* d_posX, float* d_posY, float* d_q,
    // Dane Gridu (do szybkiego wyszukiwania)
    int* d_gridParticleIndex,
    int* d_cellStart, int* d_cellEnd)
{
    // Współrzędne piksela
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // 1. Zamiana piksela na świat [-1, 1]
    // OpenGL ma (0,0) w lewym dolnym rogu
    float u = (x / (float)width) * 2.0f - 1.0f;
    float v = (y / (float)height) * 2.0f - 1.0f;

    // 2. W jakiej komórce jest ten piksel?
    int2 myGridPos = getGridPos(u, v);
    float potential = 0.0f;

    // 3. Sprawdzamy sąsiednie komórki (3x3)
    // Dzięki temu liczymy wpływ tylko bliskich cząstek!
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {

            int nx = myGridPos.x + dx;
            int ny = myGridPos.y + dy;

            // Sprawdzamy czy sąsiad jest w mapie
            if (nx >= 0 && nx < COLUMNS_COUNT && ny >= 0 && ny < ROWS_COUNT) {
                int hash = getGridHash(nx, ny);

                int startIndex = d_cellStart[hash];

                // Jeśli komórka nie jest pusta
                if (startIndex != -1) {
                    int endIndex = d_cellEnd[hash];

                    // Pętla po cząstkach w tej konkretnej komórce
                    for (int k = startIndex; k < endIndex; k++) {

                        // INDIRECTION: Pobieramy prawdziwy indeks cząstki z posortowanej mapy
                        int realIdx = d_gridParticleIndex[k];

                        // Pobieramy dane cząstki (SoA)
                        float px = d_posX[realIdx];
                        float py = d_posY[realIdx];
                        float q = d_q[realIdx];

                        // Liczymy odległość
                        float dX = u - px;
                        float dY = v - py;
                        float distSq = dX * dX + dY * dY + 0.00001f; // Epsilon

                        // Prawo Coulomba (uproszczone dla wizualizacji)
                        // Cutoff wizualny - rysujemy tylko bliskie, żeby było widać strukturę
                        if (distSq < 0.05f) {
                            potential += q / sqrtf(distSq);
                        }
                    }
                }
            }
        }
    }

    // 4. Kolorowanie (Heatmapa)
    uchar4 color = make_uchar4(0, 0, 0, 255);
    float intensity = fabs(potential) * 40.0f; // Skalowanie jasności

    if (potential > 0) {
        // Protony -> Czerwony
        color.x = clipColor(intensity);
    }
    else {
        // Elektrony -> Niebieski
        color.z = clipColor(intensity);
    }

    // Zapis do pamięci obrazu
    int pixelIndex = y * width + x;
    outputSurface[pixelIndex] = color;
}


// =================================================================================
// SEKCJA 3: HOST WRAPPER (Wywoływane z C++)
// =================================================================================

void launchSimulation(workspace* ws, float time)
{
    int numParticles = PARTICLES_COUNT;
    int numCells = COLUMNS_COUNT * ROWS_COUNT;

    // --- FAZA 1: Budowa Grida ---

    // Obliczamy Hash dla każdej cząstki
    int blocksParticles = (numParticles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calcHash << <blocksParticles, BLOCK_SIZE >> > (
        ws->d_gridParticleHash,
        ws->d_gridParticleIndex,
        ws->d_posX,
        ws->d_posY,
        numParticles
        );

    // Sortujemy cząstki według Hasha (używając biblioteki Thrust)
    // To kluczowy moment - sortujemy indeksy, a nie ciężkie dane!
    thrust::device_ptr<int> t_hash(ws->d_gridParticleHash);
    thrust::device_ptr<int> t_index(ws->d_gridParticleIndex);
    thrust::sort_by_key(t_hash, t_hash + numParticles, t_index);

    // Resetujemy i znajdujemy granice komórek
    int blocksCells = (numCells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    resetCellStartEnd << <blocksCells, BLOCK_SIZE >> > (ws->d_cellStart, ws->d_cellEnd, numCells);

    findCellBounds << <blocksParticles, BLOCK_SIZE >> > (
        ws->d_gridParticleHash,
        ws->d_cellStart,
        ws->d_cellEnd,
        numParticles
        );

    // --- FAZA 2: Fizyka ---

    updatePhysics << <blocksParticles, BLOCK_SIZE >> > (
        ws->d_posX, ws->d_posY,
        ws->d_velX, ws->d_velY,
        numParticles, DT
        );

    // --- FAZA 3: Renderowanie (Interop) ---

    // Mapujemy zasób OpenGL (PBO) żeby CUDA mogła po nim pisać
    uchar4* d_out;
    size_t numBytes;
    cudaGraphicsMapResources(1, &ws->cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_out, &numBytes, ws->cudaResource);

    // Uruchamiamy kernel wizualizacji (1 wątek na 1 piksel)
    dim3 block(16, 16);
    dim3 grid((WINDOW_WIDTH + block.x - 1) / block.x, (WINDOW_HEIGHT + block.y - 1) / block.y);

    visualizeField << <grid, block >> > (
        d_out,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        ws->d_posX, ws->d_posY, ws->d_charge,
        ws->d_gridParticleIndex,
        ws->d_cellStart, ws->d_cellEnd
        );

    // Odmapowujemy zasób (oddajemy go OpenGL)
    cudaGraphicsUnmapResources(1, &ws->cudaResource, 0);
}