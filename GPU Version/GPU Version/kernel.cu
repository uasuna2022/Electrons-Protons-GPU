#include <iostream>

// Nagłówki dodane w Kroku 1
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Szerokość i wysokość okna
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

int main()
{
    // 1. INICJALIZACJA GLFW
    glfwInit();
    // Ustawienie wersji OpenGL na 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    // Używanie profilu Core (bez starych funkcji)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // 2. TWORZENIE OKNA
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Test Okna OpenGL (w kernel.cu)", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    // Ustawienie funkcji callback dla zmiany rozmiaru okna
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // 3. ŁADOWANIE GLAD (Funkcje OpenGL)
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 4. GŁÓWNA PĘTLA RENDEROWANIA
    while (!glfwWindowShouldClose(window))
    {
        // Obsługa wejścia (np. ESC)
        processInput(window);

        // Renderowanie: Ustawienie koloru tła (ciemny błękit)
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Zamiana buforów i sprawdzanie zdarzeń
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 5. ZAMKNIĘCIE I CZYSZCZENIE
    glfwTerminate();
    return 0;
}

// Funkcja obsługująca klawisze (zamknięcie okna po naciśnięciu ESC)
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// Funkcja callback: dostosowanie widoku po zmianie rozmiaru okna
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // Mówimy OpenGL, jak duży jest obszar renderowania
    glViewport(0, 0, width, height);
}