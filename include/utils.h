#ifndef UTILS_H
#define UTILS_H

#pragma once

#include "helper_math.h"
#include "helper_host.h"
#include "helper_string.h"
#include "helper_cuda.h"

#include "shapes.h"

#include "pcg32.h"

constexpr float EPS = 0.000001f;

inline float __device__ ray_tri(
    const float3 &ro,
    const float3 &rd,
    const float3 &p0,
    const float3 &p1,
    const float3 &p2)
{

    const float3 v10 = p1 - p0;
    const float3 v20 = p2 - p0;
    const float3 h = cross(rd, v20);
    const float a = dot(v10, h);
    if (a > -1e-16f && a < 1e-16f)
    {
        // Ray is parallel to the triangle.
        return -1.f;
    }

    const float inv_a = 1.f / a;
    const float3 s = ro - p0;
    const float u = inv_a * dot(s, h);
    if (u < 0.f || u > 1.f)
    {
        return -1.f;
    }

    const float3 q = cross(s, v10);
    const float v = inv_a * dot(rd, q);
    if (v < 0.f || v > 1.f)
    {
        return -1.f;
    }

    return inv_a * dot(v20, q);
}

inline __device__ std::pair<float, float> ray_tet(
    const float3 &ro,
    const float3 &rd,
    const float3 &p0,
    const float3 &p1,
    const float3 &p2,
    const float3 &p3)
{
    // TODO: use this: http://users.uop.gr/~nplatis/files/PlatisTheoharisRayTetra.pdf
    float tmin = std::numeric_limits<float>::infinity();
    float tmax = -std::numeric_limits<float>::infinity();

    float t = ray_tri(ro, rd, p0, p1, p2);
    if (t > 0)
    {
        tmin = fminf(tmin, t);
        tmax = fmaxf(tmax, t);
    }

    t = ray_tri(ro, rd, p0, p3, p2);
    if (t > 0)
    {
        tmin = fminf(tmin, t);
        tmax = fmaxf(tmax, t);
    }

    t = ray_tri(ro, rd, p1, p2, p3);
    if (t > 0)
    {
        tmin = fminf(tmin, t);
        tmax = fmaxf(tmax, t);
    }

    t = ray_tri(ro, rd, p0, p1, p3);
    if (t > 0)
    {
        tmin = fminf(tmin, t);
        tmax = fmaxf(tmax, t);
    }

    return std::make_pair(tmin, tmax);
}

inline __device__ float det4x4_set_grad(
    const float4 &v0,
    const float4 &v1,
    const float4 &v2,
    const float4 &v3,
    float4 &dfdv0,
    float4 &dfdv1,
    float4 &dfdv2,
    float4 &dfdv3)
{
    float out = v0.w * v1.z * v2.y * v3.x - v0.z * v1.w * v2.y * v3.x -
                v0.w * v1.y * v2.z * v3.x + v0.y * v1.w * v2.z * v3.x +

                v0.z * v1.y * v2.w * v3.x - v0.y * v1.z * v2.w * v3.x -
                v0.w * v1.z * v2.x * v3.y + v0.z * v1.w * v2.x * v3.y +

                v0.w * v1.x * v2.z * v3.y - v0.x * v1.w * v2.z * v3.y -
                v0.z * v1.x * v2.w * v3.y + v0.x * v1.z * v2.w * v3.y +

                v0.w * v1.y * v2.x * v3.z - v0.y * v1.w * v2.x * v3.z -
                v0.w * v1.x * v2.y * v3.z + v0.x * v1.w * v2.y * v3.z +

                v0.y * v1.x * v2.w * v3.z - v0.x * v1.y * v2.w * v3.z -
                v0.z * v1.y * v2.x * v3.w + v0.y * v1.z * v2.x * v3.w +

                v0.z * v1.x * v2.y * v3.w - v0.x * v1.z * v2.y * v3.w -
                v0.y * v1.x * v2.z * v3.w + v0.x * v1.y * v2.z * v3.w;

    dfdv0.x = v1.w * v2.y * v3.z - v1.w * v2.z * v3.y - v1.y * v2.w * v3.z + v1.y * v2.z * v3.w + v1.z * v2.w * v3.y - v1.z * v2.y * v3.w;
    dfdv0.y = -v1.w * v2.x * v3.z + v1.w * v2.z * v3.x + v1.x * v2.w * v3.z - v1.x * v2.z * v3.w - v1.z * v2.w * v3.x + v1.z * v2.x * v3.w;
    dfdv0.z = v1.w * v2.x * v3.y - v1.w * v2.y * v3.x - v1.x * v2.w * v3.y + v1.x * v2.y * v3.w + v1.y * v2.w * v3.x - v1.y * v2.x * v3.w;
    dfdv0.w = -v1.x * v2.y * v3.z + v1.x * v2.z * v3.y + v1.y * v2.x * v3.z - v1.y * v2.z * v3.x - v1.z * v2.x * v3.y + v1.z * v2.y * v3.x;
    dfdv1.x = -v0.w * v2.y * v3.z + v0.w * v2.z * v3.y + v0.y * v2.w * v3.z - v0.y * v2.z * v3.w - v0.z * v2.w * v3.y + v0.z * v2.y * v3.w;
    dfdv1.y = v0.w * v2.x * v3.z - v0.w * v2.z * v3.x - v0.x * v2.w * v3.z + v0.x * v2.z * v3.w + v0.z * v2.w * v3.x - v0.z * v2.x * v3.w;
    dfdv1.z = -v0.w * v2.x * v3.y + v0.w * v2.y * v3.x + v0.x * v2.w * v3.y - v0.x * v2.y * v3.w - v0.y * v2.w * v3.x + v0.y * v2.x * v3.w;
    dfdv1.w = v0.x * v2.y * v3.z - v0.x * v2.z * v3.y - v0.y * v2.x * v3.z + v0.y * v2.z * v3.x + v0.z * v2.x * v3.y - v0.z * v2.y * v3.x;
    dfdv2.x = v0.w * v1.y * v3.z - v0.w * v1.z * v3.y - v0.y * v1.w * v3.z + v0.y * v1.z * v3.w + v0.z * v1.w * v3.y - v0.z * v1.y * v3.w;
    dfdv2.y = -v0.w * v1.x * v3.z + v0.w * v1.z * v3.x + v0.x * v1.w * v3.z - v0.x * v1.z * v3.w - v0.z * v1.w * v3.x + v0.z * v1.x * v3.w;
    dfdv2.z = v0.w * v1.x * v3.y - v0.w * v1.y * v3.x - v0.x * v1.w * v3.y + v0.x * v1.y * v3.w + v0.y * v1.w * v3.x - v0.y * v1.x * v3.w;
    dfdv2.w = -v0.x * v1.y * v3.z + v0.x * v1.z * v3.y + v0.y * v1.x * v3.z - v0.y * v1.z * v3.x - v0.z * v1.x * v3.y + v0.z * v1.y * v3.x;
    dfdv3.x = -v0.w * v1.y * v2.z + v0.w * v1.z * v2.y + v0.y * v1.w * v2.z - v0.y * v1.z * v2.w - v0.z * v1.w * v2.y + v0.z * v1.y * v2.w;
    dfdv3.y = v0.w * v1.x * v2.z - v0.w * v1.z * v2.x - v0.x * v1.w * v2.z + v0.x * v1.z * v2.w + v0.z * v1.w * v2.x - v0.z * v1.x * v2.w;
    dfdv3.z = -v0.w * v1.x * v2.y + v0.w * v1.y * v2.x + v0.x * v1.w * v2.y - v0.x * v1.y * v2.w - v0.y * v1.w * v2.x + v0.y * v1.x * v2.w;
    dfdv3.w = v0.x * v1.y * v2.z - v0.x * v1.z * v2.y - v0.y * v1.x * v2.z + v0.y * v1.z * v2.x + v0.z * v1.x * v2.y - v0.z * v1.y * v2.x;
    return out;
}

inline __device__ float4 tet_bary_set_grad(
    const float3 &p0_,
    const float3 &v0_,
    const float3 &v1_,
    const float3 &v2_,
    const float3 &v3_,

    float3 &db0_dp0_,
    float3 &db0_dv0_,
    float3 &db0_dv1_,
    float3 &db0_dv2_,
    float3 &db0_dv3_,
    float3 &db1_dp0_,
    float3 &db1_dv0_,
    float3 &db1_dv1_,
    float3 &db1_dv2_,
    float3 &db1_dv3_,
    float3 &db2_dp0_,
    float3 &db2_dv0_,
    float3 &db2_dv1_,
    float3 &db2_dv2_,
    float3 &db2_dv3_,
    float3 &db3_dp0_,
    float3 &db3_dv0_,
    float3 &db3_dv1_,
    float3 &db3_dv2_,
    float3 &db3_dv3_)
{

    const float4 p0 = make_float4(p0_, 1.f);
    const float4 v0 = make_float4(v0_, 1.f);
    const float4 v1 = make_float4(v1_, 1.f);
    const float4 v2 = make_float4(v2_, 1.f);
    const float4 v3 = make_float4(v3_, 1.f);

    float4 ddet0_dv0, ddet0_dv1, ddet0_dv2, ddet0_dv3;
    const float det0 = det4x4_set_grad(v0, v1, v2, v3, ddet0_dv0, ddet0_dv1, ddet0_dv2, ddet0_dv3);
    float4 ddet1_dp0, ddet1_dv1, ddet1_dv2, ddet1_dv3;
    const float det1 = det4x4_set_grad(p0, v1, v2, v3, ddet1_dp0, ddet1_dv1, ddet1_dv2, ddet1_dv3);
    float4 ddet2_dv0, ddet2_dp0, ddet2_dv2, ddet2_dv3;
    const float det2 = det4x4_set_grad(v0, p0, v2, v3, ddet2_dv0, ddet2_dp0, ddet2_dv2, ddet2_dv3);
    float4 ddet3_dv0, ddet3_dv1, ddet3_dp0, ddet3_dv3;
    const float det3 = det4x4_set_grad(v0, v1, p0, v3, ddet3_dv0, ddet3_dv1, ddet3_dp0, ddet3_dv3);
    float4 ddet4_dv0, ddet4_dv1, ddet4_dv2, ddet4_dp0;
    const float det4 = det4x4_set_grad(v0, v1, v2, p0, ddet4_dv0, ddet4_dv1, ddet4_dv2, ddet4_dp0);

    float3 ddet0_dv0_ = make_float3(ddet0_dv0.x, ddet0_dv0.y, ddet0_dv0.z);
    float3 ddet0_dv1_ = make_float3(ddet0_dv1.x, ddet0_dv1.y, ddet0_dv1.z);
    float3 ddet0_dv2_ = make_float3(ddet0_dv2.x, ddet0_dv2.y, ddet0_dv2.z);
    float3 ddet0_dv3_ = make_float3(ddet0_dv3.x, ddet0_dv3.y, ddet0_dv3.z);
    float3 ddet1_dp0_ = make_float3(ddet1_dp0.x, ddet1_dp0.y, ddet1_dp0.z);
    float3 ddet1_dv1_ = make_float3(ddet1_dv1.x, ddet1_dv1.y, ddet1_dv1.z);
    float3 ddet1_dv2_ = make_float3(ddet1_dv2.x, ddet1_dv2.y, ddet1_dv2.z);
    float3 ddet1_dv3_ = make_float3(ddet1_dv3.x, ddet1_dv3.y, ddet1_dv3.z);
    float3 ddet2_dv0_ = make_float3(ddet2_dv0.x, ddet2_dv0.y, ddet2_dv0.z);
    float3 ddet2_dp0_ = make_float3(ddet2_dp0.x, ddet2_dp0.y, ddet2_dp0.z);
    float3 ddet2_dv2_ = make_float3(ddet2_dv2.x, ddet2_dv2.y, ddet2_dv2.z);
    float3 ddet2_dv3_ = make_float3(ddet2_dv3.x, ddet2_dv3.y, ddet2_dv3.z);
    float3 ddet3_dv0_ = make_float3(ddet3_dv0.x, ddet3_dv0.y, ddet3_dv0.z);
    float3 ddet3_dv1_ = make_float3(ddet3_dv1.x, ddet3_dv1.y, ddet3_dv1.z);
    float3 ddet3_dp0_ = make_float3(ddet3_dp0.x, ddet3_dp0.y, ddet3_dp0.z);
    float3 ddet3_dv3_ = make_float3(ddet3_dv3.x, ddet3_dv3.y, ddet3_dv3.z);
    float3 ddet4_dv0_ = make_float3(ddet4_dv0.x, ddet4_dv0.y, ddet4_dv0.z);
    float3 ddet4_dv1_ = make_float3(ddet4_dv1.x, ddet4_dv1.y, ddet4_dv1.z);
    float3 ddet4_dv2_ = make_float3(ddet4_dv2.x, ddet4_dv2.y, ddet4_dv2.z);
    float3 ddet4_dp0_ = make_float3(ddet4_dp0.x, ddet4_dp0.y, ddet4_dp0.z);

    float idet0_sq_clamp = (det0 * det0 > 0 ? (1.0f / (det0 * det0)) : 0.0f);

    db0_dp0_ = (ddet1_dp0_ * det0 - 0 * det1) * idet0_sq_clamp;
    db0_dv0_ = (0 * det0 - ddet0_dv0_ * det1) * idet0_sq_clamp;
    db0_dv1_ = (ddet1_dv1_ * det0 - ddet0_dv1_ * det1) * idet0_sq_clamp;
    db0_dv2_ = (ddet1_dv2_ * det0 - ddet0_dv2_ * det1) * idet0_sq_clamp;
    db0_dv3_ = (ddet1_dv3_ * det0 - ddet0_dv3_ * det1) * idet0_sq_clamp;
    db1_dp0_ = (ddet2_dp0_ * det0 - 0 * det2) * idet0_sq_clamp;
    db1_dv0_ = (ddet2_dv0_ * det0 - ddet0_dv0_ * det2) * idet0_sq_clamp;
    db1_dv1_ = (0 * det0 - ddet0_dv1_ * det2) * idet0_sq_clamp;
    db1_dv2_ = (ddet2_dv2_ * det0 - ddet0_dv2_ * det2) * idet0_sq_clamp;
    db1_dv3_ = (ddet2_dv3_ * det0 - ddet0_dv3_ * det2) * idet0_sq_clamp;
    db2_dp0_ = (ddet3_dp0_ * det0 - 0 * det3) * idet0_sq_clamp;
    db2_dv0_ = (ddet3_dv0_ * det0 - ddet0_dv0_ * det3) * idet0_sq_clamp;
    db2_dv1_ = (ddet3_dv1_ * det0 - ddet0_dv1_ * det3) * idet0_sq_clamp;
    db2_dv2_ = (0 * det0 - ddet0_dv2_ * det3) * idet0_sq_clamp;
    db2_dv3_ = (ddet3_dv3_ * det0 - ddet0_dv3_ * det3) * idet0_sq_clamp;
    db3_dp0_ = (ddet4_dp0_ * det0 - 0 * det4) * idet0_sq_clamp;
    db3_dv0_ = (ddet4_dv0_ * det0 - ddet0_dv0_ * det4) * idet0_sq_clamp;
    db3_dv1_ = (ddet4_dv1_ * det0 - ddet0_dv1_ * det4) * idet0_sq_clamp;
    db3_dv2_ = (ddet4_dv2_ * det0 - ddet0_dv2_ * det4) * idet0_sq_clamp;
    db3_dv3_ = (0 * det0 - ddet0_dv3_ * det4) * idet0_sq_clamp;

    return make_float4(det1 / det0, det2 / det0, det3 / det0, det4 / det0);
}

inline __device__ float3 tetpos2slabuvw_bwd(
    const int &k,
    const float3 &p0,
    const float3 &v0,
    const float3 &v1,
    const float3 &v2,
    const float3 &v3,
    const float3 &uvw0,
    const float3 &uvw1,
    const float3 &uvw2,
    const float3 &uvw3,

    const float3 &grad_output,
    float3 &grad_p0,
    float3 &grad_v0,
    float3 &grad_v1,
    float3 &grad_v2,
    float3 &grad_v3)
{

    float3 db0_dp0;
    float3 db0_dv0;
    float3 db0_dv1;
    float3 db0_dv2;
    float3 db0_dv3;
    float3 db1_dp0;
    float3 db1_dv0;
    float3 db1_dv1;
    float3 db1_dv2;
    float3 db1_dv3;
    float3 db2_dp0;
    float3 db2_dv0;
    float3 db2_dv1;
    float3 db2_dv2;
    float3 db2_dv3;
    float3 db3_dp0;
    float3 db3_dv0;
    float3 db3_dv1;
    float3 db3_dv2;
    float3 db3_dv3;

    // const int tet_idx = k % 5;
    const float4 barys = tet_bary_set_grad(
        p0,
        v0,
        v1,
        v2,
        v3,
        db0_dp0,
        db0_dv0,
        db0_dv1,
        db0_dv2,
        db0_dv3,
        db1_dp0,
        db1_dv0,
        db1_dv1,
        db1_dv2,
        db1_dv3,
        db2_dp0,
        db2_dv0,
        db2_dv1,
        db2_dv2,
        db2_dv3,
        db3_dp0,
        db3_dv0,
        db3_dv1,
        db3_dv2,
        db3_dv3);

    float3 y0 = barys.x * uvw0 + barys.y * uvw1 + barys.z * uvw2 + barys.w * uvw3;

    auto &dL_dy0 = grad_output;
    float3 dy0x_dp0 = db0_dp0 * uvw0.x + db1_dp0 * uvw1.x + db2_dp0 * uvw2.x + db3_dp0 * uvw3.x;
    float3 dy0y_dp0 = db0_dp0 * uvw0.y + db1_dp0 * uvw1.y + db2_dp0 * uvw2.y + db3_dp0 * uvw3.y;
    float3 dy0z_dp0 = db0_dp0 * uvw0.z + db1_dp0 * uvw1.z + db2_dp0 * uvw2.z + db3_dp0 * uvw3.z;
    grad_p0 = dL_dy0.x * dy0x_dp0 + dL_dy0.y * dy0y_dp0 + dL_dy0.z * dy0z_dp0;

    float3 dy0x_dv0 = db0_dv0 * uvw0.x + db1_dv0 * uvw1.x + db2_dv0 * uvw2.x + db3_dv0 * uvw3.x;
    float3 dy0y_dv0 = db0_dv0 * uvw0.y + db1_dv0 * uvw1.y + db2_dv0 * uvw2.y + db3_dv0 * uvw3.y;
    float3 dy0z_dv0 = db0_dv0 * uvw0.z + db1_dv0 * uvw1.z + db2_dv0 * uvw2.z + db3_dv0 * uvw3.z;
    grad_v0 = dL_dy0.x * dy0x_dv0 + dL_dy0.y * dy0y_dv0 + dL_dy0.z * dy0z_dv0;

    float3 dy0x_dv1 = db0_dv1 * uvw0.x + db1_dv1 * uvw1.x + db2_dv1 * uvw2.x + db3_dv1 * uvw3.x;
    float3 dy0y_dv1 = db0_dv1 * uvw0.y + db1_dv1 * uvw1.y + db2_dv1 * uvw2.y + db3_dv1 * uvw3.y;
    float3 dy0z_dv1 = db0_dv1 * uvw0.z + db1_dv1 * uvw1.z + db2_dv1 * uvw2.z + db3_dv1 * uvw3.z;
    grad_v1 = dL_dy0.x * dy0x_dv1 + dL_dy0.y * dy0y_dv1 + dL_dy0.z * dy0z_dv1;

    float3 dy0x_dv2 = db0_dv2 * uvw0.x + db1_dv2 * uvw1.x + db2_dv2 * uvw2.x + db3_dv2 * uvw3.x;
    float3 dy0y_dv2 = db0_dv2 * uvw0.y + db1_dv2 * uvw1.y + db2_dv2 * uvw2.y + db3_dv2 * uvw3.y;
    float3 dy0z_dv2 = db0_dv2 * uvw0.z + db1_dv2 * uvw1.z + db2_dv2 * uvw2.z + db3_dv2 * uvw3.z;
    grad_v2 = dL_dy0.x * dy0x_dv2 + dL_dy0.y * dy0y_dv2 + dL_dy0.z * dy0z_dv2;

    float3 dy0x_dv3 = db0_dv3 * uvw0.x + db1_dv3 * uvw1.x + db2_dv3 * uvw2.x + db3_dv3 * uvw3.x;
    float3 dy0y_dv3 = db0_dv3 * uvw0.y + db1_dv3 * uvw1.y + db2_dv3 * uvw2.y + db3_dv3 * uvw3.y;
    float3 dy0z_dv3 = db0_dv3 * uvw0.z + db1_dv3 * uvw1.z + db2_dv3 * uvw2.z + db3_dv3 * uvw3.z;
    grad_v3 = dL_dy0.x * dy0x_dv3 + dL_dy0.y * dy0y_dv3 + dL_dy0.z * dy0z_dv3;

    return y0;
}
 
inline __device__ void print(const Tetra &tetra)
{
    printf("Tetra v0 [%f, %f, %f] v1 [%f, %f, %f] v2 [%f, %f, %f] v3 [%f, %f, %f]\n",
           tetra.v0.x, tetra.v0.y, tetra.v0.z,
           tetra.v1.x, tetra.v1.y, tetra.v1.z,
           tetra.v2.x, tetra.v2.y, tetra.v2.z,
           tetra.v3.x, tetra.v3.y, tetra.v3.z);
}

inline __device__ float det4x4(
    const float4 &v0,
    const float4 &v1,
    const float4 &v2,
    const float4 &v3)
{
    return v0.w * v1.z * v2.y * v3.x - v0.z * v1.w * v2.y * v3.x -
           v0.w * v1.y * v2.z * v3.x + v0.y * v1.w * v2.z * v3.x +

           v0.z * v1.y * v2.w * v3.x - v0.y * v1.z * v2.w * v3.x -
           v0.w * v1.z * v2.x * v3.y + v0.z * v1.w * v2.x * v3.y +

           v0.w * v1.x * v2.z * v3.y - v0.x * v1.w * v2.z * v3.y -
           v0.z * v1.x * v2.w * v3.y + v0.x * v1.z * v2.w * v3.y +

           v0.w * v1.y * v2.x * v3.z - v0.y * v1.w * v2.x * v3.z -
           v0.w * v1.x * v2.y * v3.z + v0.x * v1.w * v2.y * v3.z +

           v0.y * v1.x * v2.w * v3.z - v0.x * v1.y * v2.w * v3.z -
           v0.z * v1.y * v2.x * v3.w + v0.y * v1.z * v2.x * v3.w +

           v0.z * v1.x * v2.y * v3.w - v0.x * v1.z * v2.y * v3.w -
           v0.y * v1.x * v2.z * v3.w + v0.x * v1.y * v2.z * v3.w;
}

static inline __device__ float4 tet_bary(const float3 &p0_, const Tetra &tet)
{
    const float4 p0 = make_float4(p0_, 1.f);
    const float4 v0 = make_float4(tet.v0, 1.f);
    const float4 v1 = make_float4(tet.v1, 1.f);
    const float4 v2 = make_float4(tet.v2, 1.f);
    const float4 v3 = make_float4(tet.v3, 1.f);

    const float det0 = det4x4(v0, v1, v2, v3);

    const float det1 = det4x4(p0, v1, v2, v3);
    const float det2 = det4x4(v0, p0, v2, v3);
    const float det3 = det4x4(v0, v1, p0, v3);
    const float det4 = det4x4(v0, v1, v2, p0);

    return make_float4(det1 / det0, det2 / det0, det3 / det0, det4 / det0);
}

static __device__ bool inside_tetra(const Tetra &tetra, const float3 &point)
{
    float4 barys = tet_bary(point, tetra);
    return barys.x >= 0 && barys.y >= 0 && barys.z >= 0 && barys.w >= 0;
}

static inline __host__ __device__ bool same_side(
    const float3 &v1,
    const float3 &v2,
    const float3 &v3,
    const float3 &v4,
    const float3 &p)
{
    auto normal = cross(v2 - v1, v3 - v1);
    float dotV4 = dot(normal, v4 - v1);
    float dotP = dot(normal, p - v1);
    return signbit(dotV4) == signbit(dotP);
}

static __host__ __device__ bool point_in_tet(const Tetra &tetra, const float3 &p)
{
    const float3 &v1 = tetra.v0;
    const float3 &v2 = tetra.v1;
    const float3 &v3 = tetra.v2;
    const float3 &v4 = tetra.v3;

    return same_side(v1, v2, v3, v4, p) &&
           same_side(v2, v3, v4, v1, p) &&
           same_side(v3, v4, v1, v2, p) &&
           same_side(v4, v1, v2, v3, p);
}

static inline __host__ __device__ float scalar_triple_product(const float3 &a, const float3 &b, const float3 &c)
{
    return dot(a, cross(b, c));
}

static __host__ __device__ float4 barycentric(
    const float3 &p,
    const Tetra &tetra)
{
    const float3 &a = tetra.v0;
    const float3 &b = tetra.v1;
    const float3 &c = tetra.v2;
    const float3 &d = tetra.v3;

    float3 vap = p - a;
    float3 vbp = p - b;

    float3 vab = b - a;
    float3 vac = c - a;
    float3 vad = d - a;

    float3 vbc = c - b;
    float3 vbd = d - b;

    float va6 = scalar_triple_product(vbp, vbd, vbc);
    float vb6 = scalar_triple_product(vap, vac, vad);
    float vc6 = scalar_triple_product(vap, vad, vab);
    float vd6 = scalar_triple_product(vap, vab, vac);
    float v6 = 1.0 / (scalar_triple_product(vab, vac, vad));

    return make_float4(va6 * v6, vb6 * v6, vc6 * v6, vd6 * v6);
}

static inline __device__ int get_tetra_id(const int2 &start, const int2 &end, const Tetra *tetras, const float3 &mid_point)
{
    int inside = -1;
    for (int id : {start.x, start.y, end.x, end.y})
    {
        if (id != -1 && point_in_tet(tetras[id], mid_point))
        {
            inside = id;
            break;
        }
    }

    return inside;
}

static inline __device__ int get_id(const int2 &neighbours, const Tetra *tetras, const float3 &point)
{
    int inside = -1;
    for (int id : {neighbours.x, neighbours.y})
    {
        if (id != -1 && point_in_tet(tetras[id], point))
        {
            inside = id;
            break;
        }
    }

    return inside;
}

static __device__ float3 pick_tet(const Tetra &tetra, float s, float t, float u)
{
    const float3 &v0 = tetra.v0;
    const float3 &v1 = tetra.v1;
    const float3 &v2 = tetra.v2;
    const float3 &v3 = tetra.v3;

    if (s + t > 1.0)
    {
        s = 1.0 - s;
        t = 1.0 - t;
    }

    if (t + u > 1.0)
    {
        float tmp = u;
        u = 1.0 - s - t;
        t = 1.0 - tmp;
    }
    else if (s + t + u > 1.0)
    {
        float tmp = u;
        u = s + t + u - 1.0;
        s = 1 - t - tmp;
    }

    float a = 1 - s - t - u;

    return v0 * a + v1 * s + v2 * t + v3 * u;
}

#endif // UTILS_H
