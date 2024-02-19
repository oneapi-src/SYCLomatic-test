#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <dpct/fft_utils.hpp>

#include <cmath>

// Function to compare two floating-point numbers with a tolerance
int almostEqual(float a, float b, float tolerance) {
    return fabs(a - b) < tolerance;
}

// Function to compare two cufftComplex variables
int compareCufftComplex(sycl::float2 a, sycl::float2 b, float tolerance) {
    return almostEqual(a.x(), b.x(), tolerance) &&
           almostEqual(a.y(), b.y(), tolerance);
}

std::string strigifyCufftComplex(sycl::float2 a) {
    return std::string("(" + std::to_string(a.x()) + ", " +
                       std::to_string(a.y()) + ")");
}

int main() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    const int n = 8;  // Size of the input array

    // Allocate memory on the host for input and output arrays
    sycl::float2 *h_input = (sycl::float2 *)malloc(sizeof(sycl::float2) * n);
    sycl::float2 *h_output = (sycl::float2 *)malloc(sizeof(sycl::float2) * n);

    // Initialize the input array with random values
    for (int i = 0; i < n; ++i) {
        h_input[i].x() = static_cast<float>(rand()) / RAND_MAX;
        h_input[i].y() = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on the host for reference output array
    sycl::float2 *h_ref_output =
        (sycl::float2 *)malloc(sizeof(sycl::float2) * n);

    // Initialize h_ref_output with expected output
    h_ref_output[0] = sycl::float2(4.94234f, 4.77104f);
    h_ref_output[1] = sycl::float2(0.914293f, -0.261793f);
    h_ref_output[2] = sycl::float2(-0.415583f, 0.264357f);
    h_ref_output[3] = sycl::float2(0.241085f, 0.382871f);
    h_ref_output[4] = sycl::float2(-0.153554f, -1.45243f);
    h_ref_output[5] = sycl::float2(-0.421167f, -1.15111f);
    h_ref_output[6] = sycl::float2(0.0986443f, 0.210444f);
    h_ref_output[7] = sycl::float2(1.51544f, 0.391681f);

    // Allocate memory on the device (GPU)
    sycl::float2 *d_input, *d_output;
    d_input = sycl::malloc_device<sycl::float2>(n, q_ct1);
    d_output = sycl::malloc_device<sycl::float2>(n, q_ct1);

    // Copy input data from host to device
    q_ct1.memcpy(d_input, h_input, sizeof(sycl::float2) * n).wait();

    // Create a cuFFT plan
    dpct::fft::fft_engine_ptr plan;
    plan = dpct::fft::fft_engine::create(
        &q_ct1, n, dpct::fft::fft_type::complex_float_to_complex_float, 1);

    // Perform forward FFT
    plan->compute<sycl::float2, sycl::float2>(
        d_input, d_output, dpct::fft::fft_direction::forward);

    // Copy output data from device to host
    q_ct1.memcpy(h_output, d_output, sizeof(sycl::float2) * n).wait();

    // Verify the result
    bool failed = false;

    float tolerance = 5e-6f;
    for (int i = 0; i < n; ++i) {
        if (!compareCufftComplex(h_output[i], h_ref_output[i], tolerance)) {
            std::cout << "Failed: at index - " << i;
            std::cout << ": " << strigifyCufftComplex(h_output[i]);
            std::cout << " != " << strigifyCufftComplex(h_ref_output[i]);
            std::cout << std::endl;
            failed = true;
            break;
        }
    }

    if(!failed) std::cout << "Verification successful" << std::endl;

    // Destroy the cuFFT plan and free allocated memory
    dpct::fft::fft_engine::destroy(plan);
    sycl::free(d_input, q_ct1);
    sycl::free(d_output, q_ct1);
    free(h_input);
    free(h_output);

    if(failed) 
        return 1;

    return 0;
}
