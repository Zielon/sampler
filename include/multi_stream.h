#pragma once

#include <cuda.h>
#include <stack>
#include <cuda_runtime.h>

#include "helper_host.h"

struct StreamAndEvent
{
public:
    StreamAndEvent()
    {
        CUDA_CHECK_THROW(cudaStreamCreate(&m_stream));
        CUDA_CHECK_THROW(cudaEventCreate(&m_event));
    }

    ~StreamAndEvent()
    {
        if (m_stream)
        {
            // free_multi_streams(m_stream);
            // free_gpu_memory_arena(m_stream);
            cudaStreamDestroy(m_stream);
        }

        if (m_event)
        {
            cudaEventDestroy(m_event);
        }
    }

    // Only allow moving of these guys. No copying.
    StreamAndEvent &operator=(const StreamAndEvent &) = delete;
    StreamAndEvent(const StreamAndEvent &) = delete;
    StreamAndEvent &operator=(StreamAndEvent &&other)
    {
        std::swap(m_stream, other.m_stream);
        std::swap(m_event, other.m_event);
        return *this;
    }

    StreamAndEvent(StreamAndEvent &&other)
    {
        *this = std::move(other);
    }

    void wait_for(cudaEvent_t event)
    {
        CUDA_CHECK_THROW(cudaStreamWaitEvent(m_stream, event, 0));
    }

    void wait_for(cudaStream_t stream)
    {
        CUDA_CHECK_THROW(cudaEventRecord(m_event, stream));
        wait_for(m_event);
    }

    void signal(cudaStream_t stream)
    {
        CUDA_CHECK_THROW(cudaEventRecord(m_event, m_stream));
        CUDA_CHECK_THROW(cudaStreamWaitEvent(stream, m_event, 0));
    }

    cudaStream_t get()
    {
        return m_stream;
    }

private:
    cudaStream_t m_stream = {};
    cudaEvent_t m_event = {};
};
