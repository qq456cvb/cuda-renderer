#include "Vector.cuh"

__host__ __device__ Vector::Vector() :
  x(0.0), y(0.0), z(0.0) { }
__host__ __device__ Vector::Vector(float v) :
  x(v), y(v), z(v) {}
__host__ __device__ Vector::Vector(const Vector &w) :
  x(w.x), y(w.y), z(w.z) { }
__host__ __device__ Vector::Vector(float xx, float yy, float zz) :
  x(xx), y(yy), z(zz) { }

__host__ __device__ float Vector::dot(const Vector &w) const {
  return this->x * w.x + this->y * w.y + this->z * w.z;
}


__host__ __device__ Vector Vector::cross(const Vector &w) const {
  return Vector(this->y * w.z - this->z * w.y, this->z * w.x - this->x * w.z, this->x * w.y - this->y * w.x);
}

__host__ __device__ Vector Vector::operator+(float s) const {
  return Vector(this->x + s, this->y + s, this->z + s);
}

__host__ __device__ Vector Vector::operator+(const Vector &w) const {
  return Vector(this->x + w.x, this->y + w.y, this->z + w.z);
}

__host__ __device__ Vector& Vector::operator+=(float s) {
  x += s; y += s; z += s;
  return *this;
}

__host__ __device__ Vector& Vector::operator+=(const Vector &v) {
  x += v.x; y += v.y; z += v.z;
  return *this;
}

__host__ __device__ Vector Vector::operator-(float s) const {
  return Vector(this->x - s, this->y - s, this->z - s);
}

__host__ __device__ Vector Vector::operator-(const Vector &w) const {
  return Vector(this->x - w.x, this->y - w.y, this->z - w.z);
}

__host__ __device__ Vector& Vector::operator-=(float s) {
  x -= s; y -= s; z -= s;
  return *this;
}

__host__ __device__ Vector Vector::operator*(float s) const {
  return Vector(s * this->x, s * this->y, s * this->z);
}

__host__ __device__ Vector Vector::operator*(const Vector &w) const {
  return Vector(this->x * w.x, this->y * w.y, this->z * w.z);
}

__host__ __device__ Vector& Vector::operator*=(float s) {
  x *= s; y *= s; z *= s;
  return *this;
}

__host__ __device__ Vector Vector::operator/(float s) const {
  return Vector(this->x / s, this->y / s, this->z / s);
}

__host__ __device__ Vector Vector::operator/(const Vector &w) const {
  return Vector(this->x / w.x, this->y / w.y, this->z / w.z);
}

__host__ __device__ Vector& Vector::operator/=(float s) {
  x /= s; y /= s; z /= s;
  return *this;
}

__host__ __device__ float Vector::norm() const {
  return sqrtf(this->x*this->x + this->y*this->y + this->z*this->z);
}

__host__ __device__ void Vector::normalize() {
  *this /= norm();
}

__host__ __device__ Vector Vector::normalized() const {
  return *this / norm();
}

__host__ __device__ float Vector::operator[](const int &idx) {
    if (idx == 0) return x;
    if (idx == 1) return y;
    if (idx == 2) return z;
    return 0.f;
}

__host__ __device__ Vector operator-(const Vector &v) {
  return Vector(-v.x, -v.y, -v.z);
}

__host__ __device__ Vector operator*(float s, const Vector &v) {
  return v*s;
}

__host__ __device__ void coordinateSystem(const Vector &v1, Vector *v2, Vector *v3) {
  *v2 = Vector(0);
  if (v1.x == 0 && v1.x == 0)
  {
    v2->x = -v1.z;
  }
  else {
    v2->x = -v1.y;
    v2->y = v1.x;
  }
  *v3 = v1.cross(*v2);
}