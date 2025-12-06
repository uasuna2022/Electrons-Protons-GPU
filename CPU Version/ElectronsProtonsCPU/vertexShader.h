#pragma once

// Responsible for point position and color transmission
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;   // Pozition (x, y)
    layout (location = 1) in vec3 aColor; // Color (r, g, b)
    
    out vec3 fColor; // Output to shader fragment

    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        fColor = aColor;
        gl_PointSize = 3.0; // Point dimension size
    }
)";