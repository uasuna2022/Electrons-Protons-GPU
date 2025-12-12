#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <string>
#include <vector>

// Załączenie Twoich nagłówków
#include "config.h"
#include "workspace.cuh"
#include "kernel.h"
#include "fieldShaders.h"
#include "particleShaders.h" 

// Funkcja pomocnicza do sprawdzania błędów kompilacji shaderów
void checkCompileErrors(unsigned int shader, std::string type) {
    int success;
    char infoLog[1024];
    if (type != "PROGRAM") {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
    else {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            std::cerr << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
        }
    }
}

// Funkcja budująca program shaderowy z kodu źródłowego
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

    // Sprzątanie
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    return ID;
}

// Callback do zmiany rozmiaru okna
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    // ---------------------------------------
    // 1. Inicjalizacja GLFW i Okna
    // ---------------------------------------
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA N-Body Simulation", NULL, NULL);
    if (window == NULL) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // ---------------------------------------
    // 2. Inicjalizacja GLAD
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // ---------------------------------------
    // 3. Inicjalizacja Symulacji (CUDA Workspace)
    // ---------------------------------------
    // To utworzy PBO, Teksturę i zarejestruje zasoby w CUDA
    workspace sim;
    sim.Initialize();

    // ---------------------------------------
    // 4. Przygotowanie Grafiki (Field Quad)
    // ---------------------------------------

    // Kompilacja shaderów pola (tła)
    unsigned int fieldShaderID = buildShaderProgram(vertexFieldShaderSource, fragmentFieldShaderSource);

    // Definicja wierzchołków pełnoekranowego prostokąta (Quad)
    // Format: X, Y, Z (OpenGL Coords) | U, V (Texture Coords / PosCUDA)
    float quadVertices[] = {
        // pozycje            // tex coords
         1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // prawy górny
         1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // prawy dolny
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // lewy dolny
        -1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // lewy górny
    };
    unsigned int quadIndices[] = {
        0, 1, 3, // pierwszy trójkąt
        1, 2, 3  // drugi trójkąt
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

    // Atrybut 0: PosGL (vec3)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Atrybut 1: PosCUDA (vec2)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // ---------------------------------------
    // 5. Pętla Główna
    // ---------------------------------------

    // Ustawienie uniforma tekstury (zawsze slot 0)
    glUseProgram(fieldShaderID);
    glUniform1i(glGetUniformLocation(fieldShaderID, "fieldTexture"), 0);

    double lastTime = glfwGetTime();
    int nbFrames = 0;
    float totalTime = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        // Czas
        float currentTime = (float)glfwGetTime();
        float deltaTime = currentTime - totalTime; // uproszczone, dla animacji
        totalTime = currentTime;

        // --- KROK 1: Obliczenia CUDA ---
        // Uruchamiamy symulację. Kernel pisze wyniki bezpośrednio do PBO (sim.pboID).
        launchSimulation(&sim, totalTime);

        // --- KROK 2: Renderowanie Pola ---
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // A. Aktualizacja Tekstury z PBO
        // To jest kluczowy moment Interop. Kopiujemy dane z PBO (CUDA) do Tekstury (OpenGL)
        // Dzieje się to wewnątrz pamięci GPU (bardzo szybko).
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, sim.pboID);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, sim.texID);

        // Kopiowanie (ostatni parametr NULL oznacza: weź dane z podpiętego bufora PBO)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // Odpinamy PBO

        // B. Rysowanie Quada z teksturą
        glUseProgram(fieldShaderID);
        glBindVertexArray(quadVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // ---------------------------------------
        // Obsługa zdarzeń i buforów
        // ---------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Licznik FPS w tytule
        double now = glfwGetTime();
        nbFrames++;
        if (now - lastTime >= 1.0) {
            std::string title = "CUDA Electrostatics | FPS: " + std::to_string(nbFrames) +
                " | Particles: " + std::to_string(PARTICLES_COUNT);
            glfwSetWindowTitle(window, title.c_str());
            nbFrames = 0;
            lastTime += 1.0;
        }
    }

    // Sprzątanie
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteBuffers(1, &quadEBO);
    glDeleteProgram(fieldShaderID);

    glfwTerminate();
    return 0;
}