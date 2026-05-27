#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define SLICES 12
#define PQ_BITS 14
#define PQ_MAX ((1 << PQ_BITS) - 1)
#define HIST_BITS 7
#define HIST_BIAS (1 << (HIST_BITS - 1))
#define HIST_BINS ((1 << HIST_BITS) - HIST_BIAS)

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D img;

struct PeakData {
    uint frame_wg_count[SLICES];
    uint frame_wg_active[SLICES];
    uint frame_sum_pq[SLICES];
    uint frame_max_pq[SLICES];
    uint frame_hist[SLICES * HIST_BINS];
};

layout(std430, set = 0, binding = 1) buffer PeakBuffer {
    PeakData data;
};

shared uint wg_sum;
shared uint wg_max;
shared uint wg_active_count;
shared uint wg_hist[HIST_BINS];

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    uvec2 img_size = textureSize(img, 0);
    
    // SAFETY: All threads must reach the barrier. We use a boolean mask 
    // instead of an early 'return'.
    bool valid = (pos.x < img_size.x && pos.y < img_size.y);

    // 1. Initialize Shared Memory (All threads participate)
    uint local_idx = gl_LocalInvocationIndex;
    uint wg_size = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
    
    if (local_idx == 0) {
        wg_sum = 0;
        wg_max = 0;
        wg_active_count = 0;
    }
    for (uint i = local_idx; i < HIST_BINS; i += wg_size) {
        wg_hist[i] = 0;
    }
    
    // Barrier is safe because no thread returned early
    barrier();

    uint y_pq = 0;
    bool is_active = false;

    if (valid) {
        vec3 color = texture(img, vec2(pos) / vec2(img_size)).rgb;
        // Use MaxRGB for better peak detection (MoltenVK compatible)
        float luma = max(color.r, max(color.g, color.b));
        
        // Manual PQ OETF (Linear 0..1 to PQ 0..1)
        // Adjust these constants if your input scaling is different
        luma = clamp(luma, 0.0, 1.0);
        float l = pow(luma, 0.1593017578125);
        l = (0.8359375 + 18.8515625 * l) / (1.0 + 18.6875 * l);
        l = pow(l, 78.84375);
        
        y_pq = uint(l * PQ_MAX + 0.5);
        is_active = (y_pq > 0);
    }

    // 2. Subgroup Reductions
    // We use subgroupAdd(uint) which is widely supported in Metal
    uint sub_sum = subgroupAdd(y_pq);
    uint sub_max = subgroupMax(y_pq);
    uint sub_active = subgroupAdd(is_active ? 1u : 0u);

    if (subgroupElect()) {
        atomicAdd(wg_sum, sub_sum);
        atomicMax(wg_max, sub_max);
        atomicAdd(wg_active_count, sub_active);
    }

    // 3. Histogram Update
    if (valid && is_active) {
        int bin = int(y_pq >> (PQ_BITS - HIST_BITS)) - HIST_BIAS;
        if (bin >= 0 && bin < HIST_BINS) {
            atomicAdd(wg_hist[bin], 1u);
        }
    }

    barrier();

    // 4. Final Global Update
    if (local_idx == 0) {
        uint slice = (gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) % SLICES;
        atomicAdd(data.frame_wg_count[slice], 1u);
        
        if (wg_active_count > 0) {
            atomicAdd(data.frame_wg_active[slice], 1u);
            atomicAdd(data.frame_sum_pq[slice], wg_sum / wg_active_count);
            atomicMax(data.frame_max_pq[slice], wg_max);
        }
    }

    // Parallel Global Histogram Write
    uint slice_offset = ((gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x) % SLICES) * HIST_BINS;
    for (uint i = local_idx; i < HIST_BINS; i += wg_size) {
        if (wg_hist[i] > 0) {
            atomicAdd(data.frame_hist[slice_offset + i], wg_hist[i]);
        }
    }
}
