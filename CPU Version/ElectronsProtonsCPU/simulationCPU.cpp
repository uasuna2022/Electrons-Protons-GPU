#include "SimulationCPU.h"
#include "config.h"
#include "vertexShader.h"
#include "fragmentShader.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

// Class constructor
SimulationCPU::SimulationCPU(int count) {
    h_posX.resize(count);  // 'h_' stands for host
    h_posY.resize(count);
    h_velX.resize(count);
    h_velY.resize(count);
    h_charge.resize(count);

    renderData.resize(count * 5); // 5 floats for one particle (x, y, r, g, b), x,y - coords, r,g,b - color of an object

    initParticles(count);
}

// Class destructor (cleaning allocated VRAM, because C++ will not do it automatically)
SimulationCPU::~SimulationCPU() {
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
}

// Random generator of particles' position, velocity and type of charge
void SimulationCPU::initParticles(int count) {
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < count; i++) {
        float randomX = static_cast<float>(rand() % WINDOW_WIDTH);
        float randomY = static_cast<float>(rand() % WINDOW_HEIGHT);
        h_posX[i] = randomX;
        h_posY[i] = randomY;

        // Interval of velocity: [-1.0F, 1.0F]
        h_velX[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0F - 1.0F;
        h_velY[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0F - 1.0F;

        h_charge[i] = (rand() % 2 == 0) ? PROTON_CHARGE : ELECTRON_CHARGE;
    }
}

// Helper function to compile one shader (using its type and .glsl code given through shader files)
GLuint SimulationCPU::compileShader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type); 

    // Loading a source .glsl code to created shader and compiling it
    glShaderSource(shader, 1, &source, NULL);  
    glCompileShader(shader);

    // Error checking
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) 
    {
        char infoLog[1000];
        glGetShaderInfoLog(shader, 1000, NULL, infoLog);
        cerr << "Błąd kompilacji shadera:\n" << infoLog << endl;
    }
    return shader;
}

// Helper function to set up both shaders and to create a shader program
void SimulationCPU::setupShaders() {
    GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
    GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram); // Checking the compatibility between 2 loaded shaders inside a program

    // Free the memory, cause these shaders have already been copied to the program 
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// Responsible for memory allocating on VRAM and defining all instructions about how the data must be read and interpreted
void SimulationCPU::initGL() {
    setupShaders();

    // Asking for "id numbers" for our VAO and VBO, appropriate class fields are fillen 
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);


    glBindVertexArray(VAO); // All data format settings are gonna be written here (in VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO); // All operations are gonna happen on this data block (on VBO)
    glBufferData(GL_ARRAY_BUFFER, renderData.size() * sizeof(float), NULL, GL_DYNAMIC_DRAW); // VBO allocating

    // Attribute 0 (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1 (r, g, b)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// Updates particles' data 
void SimulationCPU::update() {

    // For every particle we calculate a sum of all electrostatic forces between chosen particle and all others 
    // We calculate it for 'x' and 'y' separately, as in fact we don't need the magnitude of this force, but rather 
    // its vector (length and direction). 
    for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
        float fx = 0.0F;
        float fy = 0.0F;

        for (int j = 0; j < NUMBER_OF_PARTICLES; j++) {
            if (i == j) 
                continue;

            /*  Physics logic:
                according to Couloumb's law: F = k|q1||q2|/r^2
                we don't need a magnitude of F, so we count Fx and Fy components
                using Fx = F * dx / dist, Fy = F * dy / dist
            */

            float dx = h_posX[j] - h_posX[i];
            float dy = h_posY[j] - h_posY[i];
            float distSq = dx * dx + dy * dy + 1.0F;
            float dist = sqrt(distSq);

            float forceMagnitude = -K * h_charge[i] * h_charge[j] / distSq;
            float dirX = dx / dist;
            float dirY = dy / dist;

            fx += forceMagnitude * dirX;
            fy += forceMagnitude * dirY;
        }

        // 2nd Newton's law: a = F / m -> Δv/Δt = F/m -> Δv = F*Δt/m
        // Simplifying: m = 1.0F for both electrons and protons here
        float m = (h_charge[i] > 0) ? PROTON_MASS : ELECTRON_MASS;
        h_velX[i] += fx * DT / m;
        h_velY[i] += fy * DT / m;
    }

    // Movement and reflections (here we assume, that reflection slows down a particle a bit (10% of velocity))
    for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
        h_posX[i] += h_velX[i] * DT;
        h_posY[i] += h_velY[i] * DT;

        if (h_posX[i] < 0) 
        { 
            h_posX[i] = 0; 
            h_velX[i] *= -0.9F; 
        }
        if (h_posX[i] > WINDOW_WIDTH) 
        { 
            h_posX[i] = WINDOW_WIDTH; 
            h_velX[i] *= -0.9F; 
        }
        if (h_posY[i] < 0) 
        { 
            h_posY[i] = 0; 
            h_velY[i] *= -0.9F; 
        }
        if (h_posY[i] > WINDOW_HEIGHT) 
        { 
            h_posY[i] = WINDOW_HEIGHT; 
            h_velY[i] *= -0.9F; 
        }
    }
}

void SimulationCPU::render() {
    // Packing data correctly
    for (int i = 0; i < NUMBER_OF_PARTICLES; i++) {
        int idx = i * 5;

        // OpenGL sees a screen like [-1, 1] x [-1, 1] square, so we need to recalculate the coords
        float x_openGL = (h_posX[i] / WINDOW_WIDTH) * 2.0f - 1.0f;
        float y_openGL = (h_posY[i] / WINDOW_HEIGHT) * 2.0f - 1.0f;

        renderData[idx + 0] = x_openGL;
        renderData[idx + 1] = y_openGL;

        if (h_charge[i] > 0) // Proton is light-gray
        { 
            renderData[idx + 2] = 0.8F; 
            renderData[idx + 3] = 0.8F; 
            renderData[idx + 4] = 0.8F;
        }
        else // Electron is dark-gray
        { 
            renderData[idx + 2] = 0.3F; 
            renderData[idx + 3] = 0.3F; 
            renderData[idx + 4] = 0.3F;
        }
    }

    // Transfer RAM -> VRAM
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, renderData.size() * sizeof(float), renderData.data());

    // Draw (using in-built OpenGL functions)
    glDrawArrays(GL_POINTS, 0, NUMBER_OF_PARTICLES);

    // Free memory 
    glBindVertexArray(0);
}