#pragma once

struct Triangle
{
  float3 v0{};
  float3 v1{};
  float3 v2{};

  __host__ __device__ Triangle() {}
  __host__ __device__ Triangle(float3 vertex0, float3 vertex1, float3 vertex2) : v0(vertex0), v1(vertex1), v2(vertex2){};
  __host__ __device__ Triangle(const float3 &vertex0, const float3 &vertex1, const float3 &vertex2) : v0(vertex0), v1(vertex1), v2(vertex2){};
};

struct Tetra
{
  float3 v0{};
  float3 v1{};
  float3 v2{};
  float3 v3{};

  __host__ __device__ Tetra() {}
  __host__ __device__ Tetra(float3 _v0, float3 _v1, float3 _v2, float3 _v3) : v0(_v0), v1(_v1), v2(_v2), v3(_v3){};
  __host__ __device__ Tetra(
      const float3 &_v0,
      const float3 &_v1,
      const float3 &_v2,
      const float3 &_v3) : v0(_v0), v1(_v1), v2(_v2), v3(_v3){};

  __host__ __device__ float3 centorid() const
  {
    return (v0 + v1 + v2 + v3) / 4.f;
  }
};
