from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NativePipeWireCompileTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("cc"), "a C compiler is required for the PipeWire harness")
    def test_callback_recycle_and_diagnostic_paths_are_explicit(self) -> None:
        source = (ROOT / "native/pipewire_adapter.c").read_text(encoding="utf-8")
        start = source.index("static void vr_pw_process(void *userdata)\n{")
        end = source.index("\nstatic const struct pw_stream_events", start)
        process = source[start:end]
        after_dequeue = process.split("spa_buffer = pw_buffer->buffer;", 1)[1]

        self.assertEqual(process.count("pw_stream_dequeue_buffer("), 1)
        self.assertEqual(process.count("pw_stream_queue_buffer("), 1)
        # Null context / no buffer exits happen before a buffer is owned. Every
        # early exit after a successful dequeue must return it exactly once.
        early_returns = after_dequeue.count("return;")
        recycled_exits = len(
            re.findall(r"vr_pw_return_dequeued_buffer\(context, pw_buffer\);\s*return;", after_dequeue)
        )
        self.assertEqual(recycled_exits, early_returns)
        self.assertIn("if (data->type != SPA_DATA_MemPtr)", process)
        unsupported = process.index("if (data->type != SPA_DATA_MemPtr)")
        malformed = process.index("if (data->data == NULL || chunk == NULL)")
        self.assertLess(unsupported, malformed)
        self.assertIn("context->unsupported_buffers += UINT64_C(1);", process[unsupported:malformed])
        self.assertIn("context->malformed_buffers += UINT64_C(1);", process[malformed:])

    def test_pipewire_142_headers_and_runtime_link(self) -> None:
        sysroot_value = os.environ.get("VOXREFINE_PIPEWIRE_SYSROOT")
        if not sysroot_value:
            self.skipTest("set VOXREFINE_PIPEWIRE_SYSROOT to a temporary PipeWire 1.4.2 sysroot")
        sysroot = Path(sysroot_value).resolve()
        pkgconfig_dir = sysroot / "usr/lib/x86_64-linux-gnu/pkgconfig"
        if not (pkgconfig_dir / "libpipewire-0.3.pc").is_file():
            self.skipTest(f"PipeWire pkg-config metadata not found under {sysroot}")

        env = os.environ.copy()
        env["PKG_CONFIG_PATH"] = str(pkgconfig_dir)
        env["PKG_CONFIG_SYSROOT_DIR"] = str(sysroot)
        version = subprocess.run(
            ["pkg-config", "--modversion", "libpipewire-0.3"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(version, "1.4.2")
        adapter_source = (ROOT / "native/pipewire_adapter.c").read_text(encoding="utf-8")
        harness_source = (ROOT / "native/test_pipewire_adapter.c").read_text(encoding="utf-8")
        for forbidden_call in ("pw_init(", "pw_context_new(", "pw_stream_new(", "pw_stream_connect("):
            self.assertNotIn(forbidden_call, adapter_source)
            self.assertNotIn(forbidden_call, harness_source)
        sanitizer_value = os.environ.get("VOXREFINE_PIPEWIRE_SANITIZERS", "").strip()
        sanitizer_flags = (
            [f"-fsanitize={sanitizer_value}", "-fno-omit-frame-pointer"]
            if sanitizer_value
            else []
        )
        raw_cflags = subprocess.run(
            ["pkg-config", "--cflags", "libpipewire-0.3"],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        cflags: list[str] = []
        index = 0
        while index < len(raw_cflags):
            item = raw_cflags[index]
            if item == "-I" and index + 1 < len(raw_cflags):
                cflags.extend(("-isystem", raw_cflags[index + 1]))
                index += 2
                continue
            if item.startswith("-I"):
                cflags.extend(("-isystem", item[2:]))
            else:
                cflags.append(item)
            index += 1
        runtime = Path("/lib/x86_64-linux-gnu/libpipewire-0.3.so.0")
        self.assertTrue(runtime.is_file(), f"expected host runtime not found: {runtime}")

        with tempfile.TemporaryDirectory(prefix="voxrefine-pipewire-") as temporary:
            executable = Path(temporary) / "test-pipewire-adapter"
            subprocess.run(
                [
                    shutil.which("cc") or "cc",
                    "-std=c11",
                    "-O2",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-pedantic",
                    *sanitizer_flags,
                    "-D_GNU_SOURCE",
                    *cflags,
                    "-I",
                    str(ROOT / "native"),
                    str(ROOT / "native/pipewire_adapter.c"),
                    str(ROOT / "native/live_primitives.c"),
                    str(ROOT / "native/test_pipewire_adapter.c"),
                    "-L/lib/x86_64-linux-gnu",
                    "-Wl,-rpath,/lib/x86_64-linux-gnu",
                    "-l:libpipewire-0.3.so.0",
                    "-o",
                    str(executable),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(executable)], cwd=ROOT, check=True, capture_output=True, text=True
            )
            linked = subprocess.run(
                ["ldd", str(executable)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            self.assertIn("libpipewire-0.3.so.0 => /lib/x86_64-linux-gnu/libpipewire-0.3.so.0", linked)
            self.assertNotIn("not found", linked)
        self.assertIn("PipeWire adapter boundary: OK", completed.stdout)


if __name__ == "__main__":
    unittest.main()
