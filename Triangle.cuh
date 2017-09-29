#ifndef CUDARENDERER_TRIANGLE_H
#define CUDARENDERER_TRIANGLE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include "Vector.cuh"
#include "Ray.cuh"
#include "Intersection.cuh"
#include "BBox.cuh"

struct Triangle
{
  __host__ __device__ BBox worldBound() const;
  __host__ __device__ bool intersect(const Ray &, float *, Intersection *);
  __host__ __device__ Triangle(const Vector& v1, const Vector& v2, const Vector& v3);
  __host__ __device__ Triangle();
  
  Vector center, normal;
  Vector p1, p2, p3;
};



#endif //CUDARENDERER_TRIANGLE_H