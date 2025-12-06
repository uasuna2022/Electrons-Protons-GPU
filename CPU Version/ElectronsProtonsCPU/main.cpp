#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "config.h"
#include "simulationCPU.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace std;

// Helper function to centralize a window
void centerWindow(GLFWwindow* window) {
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    int monitorX, monitorY;
    glfwGetMonitorPos(monitor, &monitorX, &monitorY);

    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    int x = monitorX + (mode->width - windowWidth) / 2;
    int y = monitorY + (mode->height - windowHeight) / 2;

    glfwSetWindowPos(window, x, y);
}

int main() {

    // GLFW library initialization
    if (!glfwInit()) 
        return -1;

    // OpenGL Version 3.3 (+ force to use VBO/VAO logic)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CPU Reference Simulation", NULL, NULL);
    if (!window) 
    { 
        glfwTerminate(); 
        return -1; 
    }

    centerWindow(window);

    glfwMakeContextCurrent(window);

    // Loading GLAD functions 
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) 
    {
        cerr << "Unable to initialize GLAD" << endl;
        return -1;
    }

    // Enabling to set a custom point size
    glEnable(GL_PROGRAM_POINT_SIZE);

    SimulationCPU simCPU(NUMBER_OF_PARTICLES);
    simCPU.initGL();

    // Main loop
    while (!glfwWindowShouldClose(window)) 
    {
        double startTime = glfwGetTime();
        simCPU.update();

        // Set a clean color
        glClearColor(0.1F, 0.1F, 0.1F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);

        simCPU.render();

        glfwSwapBuffers(window); // Double-bufforing strategy
        glfwPollEvents();
        double interval = glfwGetTime() - startTime;
        cout << interval << endl;
    }

    glfwTerminate();
    return 0;
}