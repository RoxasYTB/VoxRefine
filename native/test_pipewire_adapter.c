#include "pipewire_adapter.h"

#include <stdio.h>
#include <string.h>
#include <spa/param/audio/raw.h>
#include <spa/pod/builder.h>

static int expect(vr_pw_view_status_t actual, vr_pw_view_status_t expected,
                  const char *case_name)
{
    if (actual != expected) {
        (void)fprintf(stderr, "%s: got %d, expected %d\n", case_name,
                      (int)actual, (int)expected);
        return 0;
    }
    return 1;
}

static uint32_t next_random(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

int main(void)
{
    static const uint32_t quanta[] = {0, 1, 64, 128, 256, 512, 1024};
    float data[1024];
    unsigned char pod_storage[1024];
    struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(pod_storage, sizeof(pod_storage));
    struct spa_audio_info_raw raw_info = SPA_AUDIO_INFO_RAW_INIT(
        .format = SPA_AUDIO_FORMAT_F32, .rate = 48000, .channels = 1);
    struct spa_pod *format_pod = spa_format_audio_raw_build(
        &builder, SPA_PARAM_Format, &raw_info);
    vr_pw_audio_view_t view = {(const float *)1, 7};
    size_t capacity_frames = 99;
    struct spa_chunk published = {.offset = 0, .size = 0, .stride = 0,
                                  .flags = SPA_CHUNK_FLAG_CORRUPTED};
    const struct pw_stream_events *events;
    vr_pw_stream_context_t format_context = {0};
    int ok = 1;
    size_t quantum_index;
    uint32_t random_state = UINT32_C(0x51a7e123);
    size_t fuzz_index;

    memset(data, 0, sizeof(data));
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 64 * sizeof(float),
                                           sizeof(float), &view),
                 VR_PW_VIEW_OK, "valid 64-frame span");
    if (view.samples != data || view.frames != 64) {
        (void)fprintf(stderr, "valid view fields were not populated\n");
        ok = 0;
    }
    for (quantum_index = 0; quantum_index < sizeof(quanta) / sizeof(quanta[0]);
         ++quantum_index) {
        uint32_t frames = quanta[quantum_index];
        ok &= expect(vr_pw_validate_audio_view(
                        data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1, sizeof(data), 0,
                        frames * (uint32_t)sizeof(float), sizeof(float), &view),
                    VR_PW_VIEW_OK, "variable capture quantum");
        if (view.samples != data || view.frames != frames) {
            (void)fprintf(stderr, "variable capture quantum view is inconsistent\n");
            ok = 0;
        }
    }

    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 0, sizeof(float), &view),
                 VR_PW_VIEW_OK, "zero frames");
    if (view.samples != data || view.frames != 0) {
        (void)fprintf(stderr, "zero-frame view is inconsistent\n");
        ok = 0;
    }
    ok &= expect(vr_pw_prepare_playback_region(data, 1, sizeof(data),
                                               sizeof(data) - sizeof(float),
                                               &view, &capacity_frames),
                 VR_PW_VIEW_OK, "one-frame playback capacity");
    if (view.samples != &data[1023] || capacity_frames != 1) {
        (void)fprintf(stderr, "one-frame playback region is inconsistent\n");
        ok = 0;
    }
    ok &= expect(vr_pw_prepare_playback_region(data, 1, sizeof(data), sizeof(data),
                                               &view, &capacity_frames),
                 VR_PW_VIEW_OK, "empty playback capacity");
    if (view.samples != &data[1024] || capacity_frames != 0) {
        (void)fprintf(stderr, "empty playback region is inconsistent\n");
        ok = 0;
    }
    if (!vr_pw_publish_playback_chunk(&published, 20, 1) || published.offset != 20 ||
        published.size != sizeof(float) || published.stride != (int32_t)sizeof(float) ||
        published.flags != SPA_CHUNK_FLAG_NONE) {
        (void)fprintf(stderr, "playback chunk metadata was not published correctly\n");
        ok = 0;
    }
    if (vr_pw_publish_playback_chunk(NULL, 0, 0) ||
        vr_pw_publish_playback_chunk(&published, 0,
                                     (size_t)(UINT32_MAX / sizeof(float)) + 1)) {
        (void)fprintf(stderr, "invalid playback chunk publication accepted\n");
        ok = 0;
    }

    ok &= expect(vr_pw_validate_audio_view(data, 2, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 64 * sizeof(float),
                                           sizeof(float), &view),
                 VR_PW_VIEW_UNSUPPORTED_FORMAT, "multiple data regions");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_S16, 48000, 1,
                                           sizeof(data), 0, 64 * sizeof(float),
                                           sizeof(float), &view),
                 VR_PW_VIEW_UNSUPPORTED_FORMAT, "unsupported format");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 44100, 1,
                                           sizeof(data), 0, 64 * sizeof(float),
                                           sizeof(float), &view),
                 VR_PW_VIEW_UNSUPPORTED_FORMAT, "unsupported rate");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 2,
                                           sizeof(data), 0, 64 * sizeof(float),
                                           sizeof(float), &view),
                 VR_PW_VIEW_UNSUPPORTED_FORMAT, "unsupported channels");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 64 * sizeof(float),
                                           8, &view),
                 VR_PW_VIEW_UNSUPPORTED_LAYOUT, "unsupported stride");
    ok &= expect(vr_pw_validate_audio_view(NULL, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 4, sizeof(float), &view),
                 VR_PW_VIEW_INVALID_REGION, "null payload");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), sizeof(data) + 1, 0,
                                           sizeof(float), &view),
                 VR_PW_VIEW_INVALID_REGION, "offset beyond maxsize");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), sizeof(data) - 4, 8,
                                           sizeof(float), &view),
                 VR_PW_VIEW_INVALID_REGION, "size beyond maxsize");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           UINT32_MAX, UINT32_MAX - 3, 8,
                                           sizeof(float), &view),
                 VR_PW_VIEW_INVALID_REGION, "near uint32 offset overflow");
    ok &= expect(vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 3, sizeof(float), &view),
                 VR_PW_VIEW_INVALID_REGION, "partial sample");
    ok &= expect(vr_pw_validate_audio_view((const unsigned char *)data + 1, 1,
                                           SPA_AUDIO_FORMAT_F32, 48000, 1,
                                           sizeof(data), 0, 4, sizeof(float), &view),
                 VR_PW_VIEW_INVALID_REGION, "unaligned payload");
    if (view.samples != NULL || view.frames != 0) {
        (void)fprintf(stderr, "invalid view did not clear the destination\n");
        ok = 0;
    }
    if (vr_pw_validate_audio_view(data, 1, SPA_AUDIO_FORMAT_F32, 48000, 1,
                                  sizeof(data), 0, 4, sizeof(float), NULL) !=
        VR_PW_VIEW_INVALID_ARGUMENT) {
        (void)fprintf(stderr, "null destination accepted\n");
        ok = 0;
    }

    for (fuzz_index = 0; fuzz_index < 50000; ++fuzz_index) {
        uint32_t maxsize = next_random(&random_state) % (uint32_t)sizeof(data);
        uint32_t offset = next_random(&random_state) % ((uint32_t)sizeof(data) + 8U);
        uint32_t size = next_random(&random_state) % ((uint32_t)sizeof(data) + 8U);
        uint32_t data_count = next_random(&random_state) % 3U;
        uint32_t format = (next_random(&random_state) & 1U) != 0
                              ? SPA_AUDIO_FORMAT_F32 : SPA_AUDIO_FORMAT_S16;
        uint32_t rate = (next_random(&random_state) & 1U) != 0 ? 48000U : 44100U;
        uint32_t channels = (next_random(&random_state) & 1U) != 0 ? 1U : 2U;
        int32_t stride = (int32_t)(next_random(&random_state) % 3U) * (int32_t)sizeof(float);
        vr_pw_view_status_t status;

        view.samples = (const float *)1;
        view.frames = 99;
        status = vr_pw_validate_audio_view(data, data_count, format, rate, channels,
                                           maxsize, offset, size, stride, &view);
        if (status == VR_PW_VIEW_OK) {
            uintptr_t expected_address = (uintptr_t)data + (uintptr_t)offset;
            if (data_count != 1 || format != SPA_AUDIO_FORMAT_F32 || rate != 48000 ||
                channels != 1 || stride != (int32_t)sizeof(float) || offset > maxsize ||
                size > maxsize - offset || size % sizeof(float) != 0 ||
                (uintptr_t)view.samples != expected_address || view.frames != size / sizeof(float)) {
                (void)fprintf(stderr, "random valid region violated its bounds\n");
                ok = 0;
                break;
            }
        } else if (view.samples != NULL || view.frames != 0) {
            (void)fprintf(stderr, "random invalid region retained a borrowed pointer\n");
            ok = 0;
            break;
        }

        capacity_frames = 99;
        status = vr_pw_prepare_playback_region(data, data_count, maxsize, offset,
                                               &view, &capacity_frames);
        if (status == VR_PW_VIEW_OK) {
            if (capacity_frames != (maxsize - offset) / sizeof(float) ||
                view.frames != capacity_frames ||
                (uintptr_t)view.samples != (uintptr_t)data + (uintptr_t)offset) {
                (void)fprintf(stderr, "random playback region violated its bounds\n");
                ok = 0;
                break;
            }
        } else if (view.samples != NULL || view.frames != 0 || capacity_frames != 0) {
            (void)fprintf(stderr, "random invalid playback view was not cleared\n");
            ok = 0;
            break;
        }
    }

    events = vr_pw_stream_events();
    if (events == NULL || events->version != PW_VERSION_STREAM_EVENTS ||
        events->state_changed == NULL || events->param_changed == NULL ||
        events->process == NULL) {
        (void)fprintf(stderr, "PipeWire event table is incomplete\n");
        ok = 0;
    }
    if (pw_stream_state_as_string(PW_STREAM_STATE_UNCONNECTED) == NULL) {
        (void)fprintf(stderr, "PipeWire runtime symbol check failed\n");
        ok = 0;
    }
    if (!vr_pw_format_supported(format_pod)) {
        (void)fprintf(stderr, "supported raw format pod was rejected\n");
        ok = 0;
    }
    atomic_init(&format_context.negotiated_supported_format, false);
    events->param_changed(&format_context, SPA_PARAM_Format, format_pod);
    if (!atomic_load_explicit(&format_context.negotiated_supported_format,
                              memory_order_acquire)) {
        (void)fprintf(stderr, "supported format transition was not published\n");
        ok = 0;
    }
    raw_info.format = SPA_AUDIO_FORMAT_S16;
    builder = SPA_POD_BUILDER_INIT(pod_storage, sizeof(pod_storage));
    format_pod = spa_format_audio_raw_build(&builder, SPA_PARAM_Format, &raw_info);
    if (vr_pw_format_supported(format_pod) || vr_pw_format_supported(NULL)) {
        (void)fprintf(stderr, "unsupported raw format pod was accepted\n");
        ok = 0;
    }
    events->param_changed(&format_context, SPA_PARAM_Format, format_pod);
    if (atomic_load_explicit(&format_context.negotiated_supported_format,
                             memory_order_acquire)) {
        (void)fprintf(stderr, "unsupported format transition left old format active\n");
        ok = 0;
    }
    raw_info.format = SPA_AUDIO_FORMAT_F32;
    builder = SPA_POD_BUILDER_INIT(pod_storage, sizeof(pod_storage));
    format_pod = spa_format_audio_raw_build(&builder, SPA_PARAM_Format, &raw_info);
    events->param_changed(&format_context, SPA_PARAM_Format, format_pod);
    if (!atomic_load_explicit(&format_context.negotiated_supported_format,
                              memory_order_acquire)) {
        (void)fprintf(stderr, "supported renegotiation was not republished\n");
        ok = 0;
    }
    if (!ok) {
        return 1;
    }
    (void)puts("PipeWire adapter boundary: OK");
    return 0;
}
