// (Minimal) OpenGL to CUDA PBO example
// Maps the default OpenGL depth buffer to CUDA using GL_PIXEL_PACK_BUFFER_ARB
// Purpose of this example is to evaluate why depth transfer is so slow
// Play around with the example by commenting/uncommenting code in lines 77 ff. and in lines 110/112
//
// In order to reproduce the issue, you require:
//  - CUDA (tested with CUDA toolkit 7.5)
//  - GLEW (a version with support for GL_KHR_debug)
//  - (e.g.) freeglut (we need an OpenGL Debug context!)
//
// On Ubuntu 14.04, this example then compiles with the following command line
//  - nvcc main.cu -lglut -lGLEW -lGL
//

#include <assert.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include "Vector.cuh"
#include "Triangle.cuh"
#include "BVHAccel.cuh"

#include <cuda_gl_interop.h>

#define WIDTH  640
#define HEIGHT 480

GLuint tex, buffer;
cudaGraphicsResource_t resource = 0;
void* device_ptr = 0;
LinearBVHNode *nodes = NULL;

#define cutilSafeCall(err)  __cudaSafeCall(err,__FILE__,__LINE__)
inline void __cudaSafeCall(cudaError err,
                           const char *file, const int line){
  if(cudaSuccess != err) {
    printf("%s(%i) : cutilSafeCall() Runtime API error : %s.\n",
           file, line, cudaGetErrorString(err) );
    exit(-1);
  }
}


// Some dummy kernel to prevent optimizations
__global__ void kernel(unsigned char* pixels)
{
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 10; j++) {
            pixels[(i * WIDTH + j) * 4] = 255;
        }
    }
}

__global__ void rayTracer(int width, int height, unsigned char* pixels) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    Vector dir;
    // dir.normalize();

    // Transform t = Transform::rotateX(45.f / 180.f * CUDART_PI_F);
    // Ray ray(Point(0, 1.f, 0), t(dir), 0);
    // Vector v(0.f);

    // //renderer->setSharedNodes(scene, shared_nodes);
    // v = renderer->Li(scene, ray);

    // image[3 * (y * width + x)] = min(v.x, 1.f);
    // image[3 * (y * width + x) + 1] = min(v.y, 1.f);
    // image[3 * (y * width + x) + 2] = min(v.z, 1.f);
}

void create_buffer(GLuint* buffer)
{
    glGenBuffersARB(1, buffer);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, *buffer);
    glBufferData(GL_PIXEL_PACK_BUFFER, WIDTH*HEIGHT*4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void destroy_buffer(GLuint* buffer)
{
    glBindBuffer(GL_TEXTURE_2D, 0);
    glDeleteBuffers(1, buffer);
    *buffer = 0;
}

void create_texture(GLuint* tex)
{
    glGenTextures(1, tex);
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void destroy_texture(GLuint* tex)
{
    glBindTexture(GL_TEXTURE_2D, *tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0, 0, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, tex);
}

int initSeaMesh(Triangle **tris, int M, int N, float Lx, float Lz) {
    float dx = Lx / (M - 1);
    float dz = Lz / (N - 1);

    int tri_cnt = M * N * 2;
    *tris = new Triangle[tri_cnt];
    int idx = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++, idx += 2) {
            Vector p1(dx * i - Lx / 2, 0, dz * j - Lz / 2);
            Vector p2(dx * (i + 1) - Lx / 2, 0, dz * j - Lz / 2);
            Vector p3(dx * i - Lx / 2, 0, dz * (j + 1) - Lz / 2);
            Vector p4(dx * (i + 1) - Lx / 2, 0, dz * (j + 1) - Lz / 2);

            (*tris)[idx].p1 = p1;
            (*tris)[idx].p2 = p2;
            (*tris)[idx].p3 = p3;

            (*tris)[idx + 1].p1 = p3;
            (*tris)[idx + 1].p2 = p2;
            (*tris)[idx + 1].p3 = p4;
        }
    }
    return tri_cnt;
}

// Display function, issues gl2cuda
void display_func()
{
    glClear(GL_COLOR_BUFFER_BIT);

    size_t size = 0;
    cutilSafeCall(cudaGraphicsMapResources(1, &resource));
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, resource));
    cudaMemset(device_ptr, 0x40, WIDTH * HEIGHT * sizeof(unsigned)); 
    // kernel<<<1, 1>>>((unsigned char*)device_ptr);
    cutilSafeCall(cudaGraphicsUnmapResources(1, &resource));

    glBindTexture(GL_TEXTURE_2D, tex);

    // Readback with depth/stencil is achingly slow (alas you employ the workaround from line 77 ff.)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
    glEnd();
    glDisable(GL_TEXTURE_2D);
    // Readback of colors (for comparison) is as fast as expected
//  gl2cuda(WIDTH * sizeof(unsigned), HEIGHT, GL_BGRA, GL_UNSIGNED_BYTE);

    glutSwapBuffers();
}

int main(int argc, char** argv)
{
    glutInitWindowSize(WIDTH, HEIGHT);
    glutInit(&argc, argv);
    // Need freeglut for GLUT_DEBUG!
    glutCreateWindow("Depth readback example");

    glewInit();

    create_buffer(&buffer);
    create_texture(&tex);

    cutilSafeCall(cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsRegisterFlagsWriteDiscard));
    
    int M = 1 << 6, N = 1 << 6;
    float Lx = 2.f, Lz = 2.f;
    Triangle *tris = NULL;
    int cnt = initSeaMesh(&tris, M, N, Lx, Lz);

    BVHAccel bvh(tris, cnt);
    cutilSafeCall(cudaMalloc((void**)&nodes, bvh.node_num * sizeof(LinearBVHNode)));
    cutilSafeCall(cudaMemcpy(nodes, bvh.nodes, bvh.node_num * sizeof(LinearBVHNode), cudaMemcpyHostToDevice));
    // printf("LinearBVHNode size: %d\n", sizeof(LinearBVHNode));

    glutDisplayFunc(display_func);
    glutMainLoop();
    destroy_buffer(&buffer);
    destroy_texture(&tex);
}