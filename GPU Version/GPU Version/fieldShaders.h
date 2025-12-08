#pragma once
// Shaders for electric field

const char* vertexFieldShaderSource = R"(
	#version 330 core
	layout (location = 0) in vec3 PosGL;     // [-1,1]x[-1,1]   OpenGL coords
	layout (location = 1) in vec2 PosCUDA;   // [0,1] x [0,1]   Procent logic (i.e. 35% of width and 47% of height)
	
	out vec2 oPosCuda;
	
	void main() {
		gl_Position = vec4(PosGL, 1.0);
		oPosCuda = PosCUDA;
	}
)";

const char* fragmentFieldShaderSource = R"(
	#version 330 core
	in vec2 oPosCuda;
	out vec4 FragColor;

	uniform sampler2D fieldTexture; // Something like "2d images reader"

	void main() {
		// texture function logic: go to fieldtexture image, find oPosCuda (x,y) point, read its color 
		// and return normalized vector (i.e. 1.0 0.0 0.0 instead of 255 0 0)
		FragColor = texture(fieldTexture, oPosCuda);
	}
)";
