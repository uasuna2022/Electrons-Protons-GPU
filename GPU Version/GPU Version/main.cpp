#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>
#include <vector>

#include "config.h"
#include "workspace.cuh"
#include "kernel.h"
#include "fieldShaders.h"
#include "particleShaders.h" 

using namespace std;

/*
#ifdef _WIN32
#include <windows.h>
extern "C" {
    __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif
*/

// Helper function to check possible shaders compilation error
void checkCompileErrors(unsigned int shader, string type) {
    int success;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            cerr << "Shader compilation error of type: " << type << "\n" << infoLog << endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            cerr << "Program linking error of type: " << type << "\n" << infoLog << endl;
        }
    }
}

// Function which builds shader program from source 
unsigned int buildShaderProgram(const char* vShaderCode, const char* fShaderCode) {
    unsigned int vertex, fragment;

    // Vertex Shader
    vertex = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX");

    // Fragment Shader
    fragment = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT");

    // Shader Program
    unsigned int ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM");

    // Clean-up
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return ID;
}


int main() {
    // GLFW and window initialization
    if (!glfwInit()) {
        cerr << "Failed to initialize GLFW" << endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA N-Body Simulation", NULL, NULL);
    if (window == NULL) {
        cerr << "Failed to create GLFW window" << endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // GLAD initialization
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // Simulation initialization
    // PBOs, textures, VBOs are being created and registered with CUDA for interoperability here
    workspace sim;
    sim.Initialize();

    
    // Graphics setup
    // Shaders compilation for the background field map
    unsigned int fieldShaderID = buildShaderProgram(vertexFieldShaderSource, fragmentFieldShaderSource);

    // Vertices definition for a full-screen quad (a screen is being divided in 2 triangles)
    float quadVertices[] = {     
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // Top-right
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // Bottom-right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // Bottom-left
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // Top-left
    };
    unsigned int quadIndices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    unsigned int quadVAO, quadVBO, quadEBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glGenBuffers(1, &quadEBO);

    glBindVertexArray(quadVAO);

    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIndices), quadIndices, GL_STATIC_DRAW);

    // Attribute 0 - position (x,y,z)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1 - texture coordinates (u, v)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // Graphics setup
    // Shaders compliation for particles
    unsigned int particleShaderID = buildShaderProgram(vertexParticleShaderSource, fragmentParticleShaderSource);
    unsigned int particleVAO;
    glGenVertexArrays(1, &particleVAO);
    glBindVertexArray(particleVAO);

    // Bind the VBO created by CUDA (allows OpenGL to read particle positions updated by CUDA without CPU copying)
    glBindBuffer(GL_ARRAY_BUFFER, sim.particleVBO);

    // Attribute 0: position (x, y)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1: color (r, g, b)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    
    // Main loop
    glUseProgram(fieldShaderID);
    glUniform1i(glGetUniformLocation(fieldShaderID, "fieldTexture"), 0);


    glEnable(GL_PROGRAM_POINT_SIZE); // Enable shader to change point size
    glDisable(GL_DEPTH_TEST);        

    double lastTime = glfwGetTime();
    int nbFrames = 0;
    float totalTime = 0.0F;

    bool isPaused = false;
    bool spacePressed = false;

    // If space button is pressed, simulation is being stopped
    while (!glfwWindowShouldClose(window)) {
        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - totalTime;
        totalTime = currentTime;

        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            if (!spacePressed) {
                isPaused = !isPaused; 
                spacePressed = true;  

                if (isPaused) cout << "Simulation paused" << endl;
                else cout << "Simulation resumed" << endl;
            }
        }
        else spacePressed = false;
        


        // 1st step - CUDA calculation: runs kernels to update phycics, field map and graphics buffers
        if (!isPaused)
            launchSimulation(&sim, totalTime);

        // 2nd step - rendering
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // 2.1 - Background draw
        // Copy data from CUDA PBO to OPenGL Texture 
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sim.pboID);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sim.texID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        // Draw the full screen quad with the field texture
        glUseProgram(fieldShaderID);
        glBindVertexArray(quadVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);

        // 2.2 - Particles draw
        // Draw points directly using data from sim.particleVBO shared with CUDA
        glUseProgram(particleShaderID);
        glBindVertexArray(particleVAO);
        glDrawArrays(GL_POINTS, 0, PARTICLES_COUNT);
        glBindVertexArray(0);

        // Events handle
        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Print the info about FPS 
        double now = glfwGetTime();
        nbFrames++;
        if (now - lastTime >= 1.0) {
            string title = "CUDA Electrostatics | FPS: " + to_string(nbFrames) + " | Particles: " + to_string(PARTICLES_COUNT);
            glfwSetWindowTitle(window, title.c_str());
            nbFrames = 0;
            lastTime += 1.0;
        }
    }

    // Cleaning
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &quadEBO);
    glDeleteProgram(fieldShaderID);

    glDeleteVertexArrays(1, &particleVAO);
    glDeleteProgram(particleShaderID);

    glfwTerminate();
    return 0;
}