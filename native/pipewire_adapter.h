#ifndef VOXREFINE_PIPEWIRE_ADAPTER_H
#define VOXREFINE_PIPEWIRE_ADAPTER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
#include <sys/types.h>

#include <pipewire/stream.h>
#include <spa/param/audio/raw-utils.h>

#include "live_primitives.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VR_PW_VIEW_OK = 0,
    VR_PW_VIEW_INVALID_ARGUMENT,
    VR_PW_VIEW_UNSUPPORTED_FORMAT,
    VR_PW_VIEW_UNSUPPORTED_LAYOUT,
    VR_PW_VIEW_INVALID_REGION
} vr_pw_view_status_t;

/* Borrowed view: valid only as long as the caller's PipeWire buffer is held. */
typedef struct {
    const float *samples;
    size_t frames;
} vr_pw_audio_view_t;

typedef enum {
    VR_PW_CAPTURE = 0,
    VR_PW_PLAYBACK = 1
} vr_pw_direction_t;

/* Caller-owned state passed as listener userdata; init never creates a stream. */
typedef struct {
    struct pw_stream *stream;
    vr_pw_direction_t direction;
    vr_spsc_ring_t *ring;
    vr_accumulator_t *capture_accumulator;
    vr_splitter_t *playback_splitter;
    uint64_t next_capture_sequence;
    uint64_t unsupported_buffers;
    uint64_t malformed_buffers;
    uint64_t format_not_ready;
    uint64_t no_buffer_callbacks;
    /* Queue and immediate-return failures after a successful dequeue. */
    uint64_t queue_errors;
    uint64_t capture_dropped_hops;
    uint64_t playback_shortfall_frames;
    _Atomic bool negotiated_supported_format;
    _Atomic int state;
} vr_pw_stream_context_t;

bool vr_pw_capture_context_init(vr_pw_stream_context_t *context,
                                struct pw_stream *stream, vr_spsc_ring_t *ring,
                                vr_accumulator_t *accumulator);
bool vr_pw_playback_context_init(vr_pw_stream_context_t *context,
                                 struct pw_stream *stream, vr_spsc_ring_t *ring,
                                 vr_splitter_t *splitter);
bool vr_pw_format_supported(const struct spa_pod *param);

/* Validate a single interleaved mono F32 48 kHz SPA region without retaining it. */
vr_pw_view_status_t vr_pw_validate_audio_view(
    const void *data, uint32_t data_count, uint32_t format, uint32_t rate,
    uint32_t channels, uint32_t maxsize, uint32_t offset, uint32_t size,
    int32_t stride, vr_pw_audio_view_t *view);
vr_pw_view_status_t vr_pw_prepare_playback_region(
    void *data, uint32_t data_count, uint32_t maxsize, uint32_t offset,
    vr_pw_audio_view_t *view, size_t *capacity_frames);
bool vr_pw_publish_playback_chunk(struct spa_chunk *chunk, uint32_t offset,
                                  size_t frames);

/* Event table uses PipeWire 1.4-compatible callback signatures. */
const struct pw_stream_events *vr_pw_stream_events(void);

#ifdef __cplusplus
}
#endif

#endif
