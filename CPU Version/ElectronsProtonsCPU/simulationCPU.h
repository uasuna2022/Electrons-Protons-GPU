#pragma once
/*  
	simulationCPU header file, contains a list of fields and methods of simulationCPU class.
	This class is defined to show how the program runs on CPU and how long does it take CPU
	to finish one whole loop of computations for all the particles and a render process.
*/

#include "glad/glad.h"    
#include "GLFW/glfw3.h"
#include <vector>

using namespace std;

class SimulationCPU {
private:
	GLuint VAO;
	GLuint VBO;
	GLuint shaderProgram;
	vector<float> renderData;
	GLuint compileShader(const char* source, GLenum type);
	void setupShaders();
public:
	vector<float> h_posX;
	vector<float> h_posY;
	vector<float> h_velX;
	vector<float> h_velY;
	vector<float> h_charge; 

	SimulationCPU(int particlesCount);
	~SimulationCPU();

	void initParticles(int particlesCount);
	void initGL();
	void update();
	void render();
};