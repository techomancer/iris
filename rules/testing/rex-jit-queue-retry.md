# Retry rejected REX JIT compile requests

The compile channel holds 256 requests. If `try_send` fails, remove the shader
key from `queued`; otherwise subsequent draws see a phantom queued entry and
never request compilation again. A 263-entry warm-up profile reproduced this
failure and 30-second waits in graphics tests.

Unit tests must start without loading the user's persistent REX profile. Test
warm-up explicitly with fixtures rather than depending on a host's `~/.iris`
state. `full_compile_queue_allows_retry` verifies saturation and retry without
starting the real compiler.
