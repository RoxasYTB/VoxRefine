#include "pipewire_adapter.h"

#include <stdalign.h>
#include <string.h>

static void vr_pw_state_changed(void *userdata, enum pw_stream_state old,
                                enum pw_stream_state state, const char *error);
static void vr_pw_param_changed(void *userdata, uint32_t id, const struct spa_pod *param);
static void vr_pw_process(void *userdata);
static void vr_pw_return_dequeued_buffer(vr_pw_stream_context_t *context,
                                         struct pw_buffer *buffer);

bool vr_pw_capture_context_init(vr_pw_stream_context_t *context,
                                struct pw_stream *stream, vr_spsc_ring_t *ring,
                                vr_accumulator_t *accumulator)
{
    if (context == NULL || stream == NULL || ring == NULL || accumulator == NULL) {
        return false;
    }
    context->unsupported_buffers = 0;
    context->malformed_buffers = 0;
    context->format_not_ready = 0;
    context->no_buffer_callbacks = 0;
    context->queue_errors = 0;
    context->capture_dropped_hops = 0;
    context->playback_shortfall_frames = 0;
    context->next_capture_sequence = 0;
    context->stream = stream;
    context->direction = VR_PW_CAPTURE;
    context->ring = ring;
    context->capture_accumulator = accumulator;
    context->playback_splitter = NULL;
    atomic_init(&context->negotiated_supported_format, false);
    atomic_init(&context->state, PW_STREAM_STATE_UNCONNECTED);
    vr_acc_reset(accumulator);
    return true;
}

bool vr_pw_playback_context_init(vr_pw_stream_context_t *context,
                                 struct pw_stream *stream, vr_spsc_ring_t *ring,
                                 vr_splitter_t *splitter)
{
    if (context == NULL || stream == NULL || ring == NULL || splitter == NULL) {
        return false;
    }
    context->unsupported_buffers = 0;
    context->malformed_buffers = 0;
    context->format_not_ready = 0;
    context->no_buffer_callbacks = 0;
    context->queue_errors = 0;
    context->capture_dropped_hops = 0;
    context->playback_shortfall_frames = 0;
    context->next_capture_sequence = 0;
    context->stream = stream;
    context->direction = VR_PW_PLAYBACK;
    context->ring = ring;
    context->capture_accumulator = NULL;
    context->playback_splitter = splitter;
    atomic_init(&context->negotiated_supported_format, false);
    atomic_init(&context->state, PW_STREAM_STATE_UNCONNECTED);
    vr_split_reset(splitter);
    return true;
}

vr_pw_view_status_t vr_pw_validate_audio_view(
    const void *data, uint32_t data_count, uint32_t format, uint32_t rate,
    uint32_t channels, uint32_t maxsize, uint32_t offset, uint32_t size,
    int32_t stride, vr_pw_audio_view_t *view)
{
    uintptr_t address;

    if (view == NULL) {
        return VR_PW_VIEW_INVALID_ARGUMENT;
    }
    view->samples = NULL;
    view->frames = 0;

    if (data_count != 1 || format != SPA_AUDIO_FORMAT_F32 || rate != 48000 || channels != 1) {
        return VR_PW_VIEW_UNSUPPORTED_FORMAT;
    }
    if (stride != (int32_t)sizeof(float)) {
        return VR_PW_VIEW_UNSUPPORTED_LAYOUT;
    }
    if (data == NULL || offset > maxsize || size > maxsize - offset ||
        size % (uint32_t)sizeof(float) != 0) {
        return VR_PW_VIEW_INVALID_REGION;
    }

    address = (uintptr_t)data + (uintptr_t)offset;
    if (address < (uintptr_t)data || address % alignof(float) != 0) {
        return VR_PW_VIEW_INVALID_REGION;
    }

    view->samples = (const float *)address;
    view->frames = (size_t)(size / (uint32_t)sizeof(float));
    return VR_PW_VIEW_OK;
}

vr_pw_view_status_t vr_pw_prepare_playback_region(
    void *data, uint32_t data_count, uint32_t maxsize, uint32_t offset,
    vr_pw_audio_view_t *view, size_t *capacity_frames)
{
    size_t capacity;
    vr_pw_view_status_t status;

    if (view == NULL || capacity_frames == NULL) {
        return VR_PW_VIEW_INVALID_ARGUMENT;
    }
    view->samples = NULL;
    view->frames = 0;
    *capacity_frames = 0;
    if (offset > maxsize) {
        return VR_PW_VIEW_INVALID_REGION;
    }
    capacity = (size_t)((maxsize - offset) / (uint32_t)sizeof(float));
    status = vr_pw_validate_audio_view(
        data, data_count, SPA_AUDIO_FORMAT_F32, 48000, 1, maxsize, offset,
        (uint32_t)(capacity * sizeof(float)), (int32_t)sizeof(float), view);
    if (status == VR_PW_VIEW_OK) {
        *capacity_frames = capacity;
    }
    return status;
}

bool vr_pw_publish_playback_chunk(struct spa_chunk *chunk, uint32_t offset,
                                  size_t frames)
{
    if (chunk == NULL || frames > (size_t)(UINT32_MAX / sizeof(float))) {
        return false;
    }
    chunk->offset = offset;
    chunk->size = (uint32_t)(frames * sizeof(float));
    chunk->stride = (int32_t)sizeof(float);
    chunk->flags = SPA_CHUNK_FLAG_NONE;
    return true;
}

bool vr_pw_format_supported(const struct spa_pod *param)
{
    struct spa_audio_info_raw info;

    if (param == NULL) {
        return false;
    }
    memset(&info, 0, sizeof(info));
    return spa_format_audio_raw_parse(param, &info) >= 0 &&
           info.format == SPA_AUDIO_FORMAT_F32 && info.rate == 48000 && info.channels == 1;
}

static void vr_pw_state_changed(void *userdata, enum pw_stream_state old,
                                enum pw_stream_state state, const char *error)
{
    vr_pw_stream_context_t *context = userdata;

    (void)old;
    (void)error;
    if (context != NULL) {
        atomic_store_explicit(&context->state, (int)state, memory_order_relaxed);
    }
}

static void vr_pw_param_changed(void *userdata, uint32_t id, const struct spa_pod *param)
{
    vr_pw_stream_context_t *context = userdata;

    if (context == NULL || id != SPA_PARAM_Format) {
        return;
    }
    atomic_store_explicit(&context->negotiated_supported_format, false, memory_order_release);
    if (param == NULL) {
        return;
    }
    if (vr_pw_format_supported(param)) {
        atomic_store_explicit(&context->negotiated_supported_format, true, memory_order_release);
    }
}

static void vr_pw_return_dequeued_buffer(vr_pw_stream_context_t *context,
                                         struct pw_buffer *buffer)
{
    if (pw_stream_return_buffer(context->stream, buffer) < 0) {
        context->queue_errors += UINT64_C(1);
    }
}

static void vr_pw_process(void *userdata)
{
    vr_pw_stream_context_t *context = userdata;
    struct pw_buffer *pw_buffer;
    struct spa_buffer *spa_buffer;
    struct spa_data *data;
    struct spa_chunk *chunk;

    if (context == NULL || context->stream == NULL) {
        return;
    }
    pw_buffer = pw_stream_dequeue_buffer(context->stream);
    if (pw_buffer == NULL) {
        context->no_buffer_callbacks += UINT64_C(1);
        return;
    }
    spa_buffer = pw_buffer->buffer;
    if (!atomic_load_explicit(&context->negotiated_supported_format, memory_order_acquire)) {
        context->format_not_ready += UINT64_C(1);
        vr_pw_return_dequeued_buffer(context, pw_buffer);
        return;
    }
    if (spa_buffer == NULL || spa_buffer->n_datas != 1 || spa_buffer->datas == NULL) {
        context->unsupported_buffers += UINT64_C(1);
        vr_pw_return_dequeued_buffer(context, pw_buffer);
        return;
    }
    data = &spa_buffer->datas[0];
    chunk = data->chunk;
    if (data->type != SPA_DATA_MemPtr) {
        context->unsupported_buffers += UINT64_C(1);
        vr_pw_return_dequeued_buffer(context, pw_buffer);
        return;
    }
    if (data->data == NULL || chunk == NULL) {
        context->malformed_buffers += UINT64_C(1);
        vr_pw_return_dequeued_buffer(context, pw_buffer);
        return;
    }

    if (context->direction == VR_PW_CAPTURE) {
        vr_pw_audio_view_t view;
        vr_pw_view_status_t status;
        size_t position = 0;

        status = vr_pw_validate_audio_view(
            data->data, spa_buffer->n_datas, SPA_AUDIO_FORMAT_F32, 48000, 1,
            data->maxsize, chunk->offset, chunk->size, chunk->stride, &view);
        if (status == VR_PW_VIEW_UNSUPPORTED_LAYOUT ||
            (data->flags & SPA_DATA_FLAG_READABLE) == 0) {
            context->unsupported_buffers += UINT64_C(1);
            vr_pw_return_dequeued_buffer(context, pw_buffer);
            return;
        }
        if (status != VR_PW_VIEW_OK || (chunk->flags & SPA_CHUNK_FLAG_CORRUPTED) != 0 ||
            context->capture_accumulator == NULL || context->ring == NULL) {
            context->malformed_buffers += UINT64_C(1);
            vr_pw_return_dequeued_buffer(context, pw_buffer);
            return;
        }
        while (position < view.frames) {
            size_t consumed = vr_acc_push(context->capture_accumulator,
                                          view.samples + position,
                                          view.frames - position);
            position += consumed;
            if (context->capture_accumulator->buffered_frames == VR_HOP_FRAMES) {
                vr_audio_block_t block;
                if (vr_acc_pop_hop(context->capture_accumulator, block.samples)) {
                    block.sequence = context->next_capture_sequence++;
                    if (!vr_spsc_try_push(context->ring, &block)) {
                        context->capture_dropped_hops += UINT64_C(1);
                    }
                }
            } else if (consumed == 0) {
                break;
            }
        }
    } else {
        vr_pw_audio_view_t output_view;
        size_t capacity_frames = 0;
        size_t requested_frames;
        size_t rendered;

        if ((data->flags & SPA_DATA_FLAG_WRITABLE) == 0 ||
            context->playback_splitter == NULL || context->ring == NULL ||
            chunk->offset > data->maxsize) {
            context->malformed_buffers += UINT64_C(1);
            vr_pw_return_dequeued_buffer(context, pw_buffer);
            return;
        }
        if (vr_pw_prepare_playback_region(data->data, spa_buffer->n_datas,
                                          data->maxsize, chunk->offset,
                                          &output_view, &capacity_frames) != VR_PW_VIEW_OK) {
            context->malformed_buffers += UINT64_C(1);
            vr_pw_return_dequeued_buffer(context, pw_buffer);
            return;
        }
        requested_frames = capacity_frames;
        if (pw_buffer->requested != 0) {
            if (pw_buffer->requested > (uint64_t)capacity_frames) {
                context->malformed_buffers += UINT64_C(1);
                vr_pw_return_dequeued_buffer(context, pw_buffer);
                return;
            }
            requested_frames = (size_t)pw_buffer->requested;
        }
        if (!vr_pw_publish_playback_chunk(chunk, chunk->offset, requested_frames)) {
            context->malformed_buffers += UINT64_C(1);
            vr_pw_return_dequeued_buffer(context, pw_buffer);
            return;
        }
        rendered = vr_split_render(context->playback_splitter, context->ring,
                                   (float *)output_view.samples, requested_frames);
        if (rendered < requested_frames) {
            size_t missing = requested_frames - rendered;
            memset((float *)output_view.samples + rendered,
                   0, missing * sizeof(float));
            context->playback_shortfall_frames += (uint64_t)missing;
        }
    }
    if (pw_stream_queue_buffer(context->stream, pw_buffer) < 0) {
        context->queue_errors += UINT64_C(1);
    }
}

static const struct pw_stream_events vr_pw_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = vr_pw_state_changed,
    .param_changed = vr_pw_param_changed,
    .process = vr_pw_process,
};

const struct pw_stream_events *vr_pw_stream_events(void)
{
    return &vr_pw_events;
}
