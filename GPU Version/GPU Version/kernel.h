#pragma once
#include "cuda_runtime.h"

class workspace;

void launchSimulation(workspace* ws, float time);