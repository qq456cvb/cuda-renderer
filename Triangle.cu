 #include "Triangle.cuh"

 __host__ __device__ Triangle::Triangle() {}

 __host__ __device__ Triangle::Triangle(const Vector& v1, const Vector& v2, const Vector& v3):
    p1(v1), p2(v2), p3(v3) {
    center = (v1 + v2 + v3) / 3.f;
    Vector e1 = p2 - p1;
    Vector e2 = p3 - p1;
    normal = e1.cross(e2);
 }

 __host__ __device__ BBox Triangle::worldBound() const {
    return unionBox(BBox(p1, p2), p3);
}

__host__ __device__ bool Triangle::intersect(const Ray &ray, float *t_hit, Intersection *isect) {
    Vector e1 = p2 - p1;
    Vector e2 = p3 - p1;
    Vector s1 = ray.d.cross(e2);
    
    float divisor = s1.dot(e1);
    
    if (divisor == 0)
    {
      return false;
    }
    float inv_divisor = 1.f / divisor;
  
    // Compute first barycentric coordinate
    Vector s = ray.o - p1;
    float b1 = s.dot(s1) * inv_divisor;
    if (b1 < 0. || b1 > 1.)
      return false;
  
    // Compute second barycentric coordinate
    Vector s2 = s.cross(e1);
    float b2 = ray.d.dot(s2) * inv_divisor;
    if (b2 < 0. || b1 + b2 > 1.)
      return false;
  
    // Compute _t_ to intersection point
    float t = e2.dot(s2) * inv_divisor;
    //printf("t: %f\n", t);
    if (t < ray.t_min || t > ray.t_max)
      return false;

    *t_hit = t;
    isect->normal = this->normal;
    isect->p = ray(t);
    return true;
}