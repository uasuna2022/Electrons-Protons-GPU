#pragma once
// Shaders for particles  

// Responsible for setting a point position and color transmission
const char* vertexParticleShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;   // Pozition (x, y)
    layout (location = 1) in vec3 aColor; // Color (r, g, b)
    
    out vec3 fColor; // Output to fragment shader

    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0); // 2D -> 4D (required by OpenGL)
        fColor = aColor;
        gl_PointSize = 3.0; // Point dimension size
    }
)";

// Responsible for pixel coloring (its direct setting on the window)
const char* fragmentParticleShaderSource = R"(
    #version 330 core
    in vec3 fColor;
    out vec4 FragColor;

    void main() {
        FragColor = vec4(fColor, 1.0);
    }
)";