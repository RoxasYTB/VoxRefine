# PipeWire Binding Compile Check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Compile and link a minimal VoxRefine adapter against the host's exact PipeWire 1.4.2 API without creating a PipeWire context, stream, or device connection.

**Architecture:** Keep `native/live_primitives.{h,c}` device-agnostic. Add a narrow PipeWire-facing layer that defines actual event callback signatures, parses negotiated raw-audio parameters, and translates borrowed SPA data/chunk metadata into validated mono-float spans before copying into VoxRefine-owned rings. Tests call only the pure field validator and inspect the event tables; they never create fake PipeWire streams or connect to a server.

**Tech Stack:** PipeWire 1.4.2 and SPA 0.2 development headers extracted from Debian packages into a temporary sysroot; C11; `pkg-config`; host PipeWire runtime for a link-only smoke.

## Global Constraints

- Do not run `apt install`, modify system packages, call `pw_init`, create `pw_context`/`pw_stream`, or call `pw_stream_connect` in this tranche.
- Use the actual PipeWire/SPA headers and `.pc` files from version 1.4.2; do not vendor or hand-copy declarations.
- Do not assume planar/interleaved layouts, multiple data chunks, mmap, requested-size semantics, or audio-buffer sizes beyond what the negotiated 48 kHz mono F32 format and actual metadata establish.
- Borrowed PipeWire pointers may be read/copied only during a process callback and must never be retained.
- Keep sequence/drop/session policy outside the binding; keep device-side malformed-buffer/xrun counters separate from engine counters.
- The result is a compile/link and pure metadata-boundary check only; it is not a working live backend or a hardware test.

---

### Task 1: Exact-version SDK setup and compile/link check

**Files:**
- Create: `tests/test_pipewire_native.py`
- Create: `native/test_pipewire_adapter.c`

**Interfaces:**
- Tests read `VOXREFINE_PIPEWIRE_SYSROOT`; when absent, they skip with a clear reason.
- `pkg-config` uses `<sysroot>/usr/lib/x86_64-linux-gnu/pkgconfig` plus `PKG_CONFIG_SYSROOT_DIR=<sysroot>`.
- Compile with the sysroot C flags and `-std=c11 -Wall -Wextra -Werror -pedantic`; link against the installed versioned `libpipewire-0.3.so.0` without initializing PipeWire.

- [x] Verify the `.pc` version is exactly 1.4.2 and compile/link a smoke executable that reads callback-table versions and parses a synthetic SPA raw-format pod.
- [x] Verify `ldd` resolves the expected system `libpipewire-0.3.so.0`; fail if unresolved.

### Task 2: Narrow PipeWire event and buffer adapter

**Files:**
- Create: `native/pipewire_adapter.h`
- Create: `native/pipewire_adapter.c`
- Modify: `native/test_pipewire_adapter.c`

**Interfaces:**
- `vr_pw_audio_view_t` is an ephemeral read-only mono-float span `{samples, frames}`.
- `vr_pw_validate_audio_view(...)` validates scalar metadata: negotiated F32/48 kHz/mono, one data chunk, non-null/aligned payload, unit-sample stride, byte-boundary alignment, and `offset + size <= maxsize`; invalid input returns a stable enum and leaves the destination view zeroed.
- `vr_pw_stream_events()` returns an event table using the real 1.4.2 `.process`, `.state_changed`, and `.param_changed` callback types. A caller-owned context selects capture or playback. No function constructs a stream or connects it.
- The callback adapter validates SPA metadata, copies capture into the existing accumulator/ring, renders playback from the existing splitter/ring, and counts invalid device buffers and playback shortfalls separately.

- [x] Implement negotiated-format checks through the official SPA raw-audio parser.
- [x] Implement borrowed-region validation and copy-only callback integration; do not store SPA pointers after callback return.
- [x] Cover unsupported format, null payload, invalid stride, misalignment, out-of-range/near-wrap offset and size, zero frames, varying quanta, and playback capacity/publication boundaries using scalar metadata.
- [x] Confirm the PipeWire event table exposes the actual callback signatures; no stream/context creation or connect function is called.

### Task 3: Documentation and verification

**Files:**
- Create: `docs/architecture/native-pipewire-binding-2026-10-09.md`
- Modify: `docs/architecture/pipewire-adapter-contract.md`
- Modify: `README.md`

- [x] Document temporary sysroot extraction, compiler/pkg-config/link commands, callback boundary, and exact limits of the proof.
- [x] Run the SDK-enabled native test, full unit suite, compileall, ASan/UBSan for the pure boundary harness, and `git diff --check`.


## Dernière revue ciblée — 2026-10-09

Le retour GPT Web recommande de fermer cette couche jusqu’à un futur stream autorisé, après test des diagnostics et de l’ownership.

- [x] Exercé la callback `param_changed` sur format valide → invalide → valide; chaque lecture acquiert le booléen et l’ancien format n’est pas conservé après une négociation invalide.
- [x] Exercé la remise à zéro de `spa_chunk.flags` quand un buffer playback réutilisé portait `SPA_CHUNK_FLAG_CORRUPTED`.
- [x] Vérifié structurellement : une fonction de dequeue, un `queue_buffer` sur le chemin réussi et une restitution immédiate sur chaque retour anticipé après dequeue; les erreurs de queue/return sont comptées.
- [x] Vérifié que les autres types `spa_data` sont rapportés comme non supportés avant la validation des bornes qui ne concerne que MemPtr.
- [x] ASan/UBSan PipeWire 1.4.2 : 2 tests OK; tests ciblés natif/live : 15 OK; suite complète : 127 OK, 12 ignorés optionnels.

**Checkpoint clos pour cette tranche :** compilation/link contre PipeWire 1.4.2 et tests de métadonnées; aucune preuve de livraison de buffers, d’ownership interne, de cadence, d’xrun ou de temps réel avec un serveur réel. Pas de création de stream ou accès au microphone. Le prochain gain de preuve nécessite un smoke matériel demandé explicitement.
