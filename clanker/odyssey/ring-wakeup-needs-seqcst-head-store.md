# A ring's producer must publish its head with a SeqCst store

Status: fixed 2026-11-03.

**Symptom.** With the raster thread keeping quads in flight (aec14e3c), the
stress test `the_frame_ends_as_a_whole_resolve_under_live_writers` hung now
and then under the full suite: the front end waited in `raster_wait`, the
raster thread polled for the pipeline to run dry, and the shader and back
end both slept, one of them with an item still queued.

**Cause.** The shader's and back end's rings use the sleep handshake: the
consumer stores `sleeping = true` and then loads the head; the producer
stores the head and then loads `sleeping`, and kicks the consumer if it is
set. The producer stored the head with `Release`. On x86 a later load may
pass an earlier plain store, so the producer could read `sleeping` as false
before its head store was visible, while the consumer read the old head
and went to sleep. Nobody kicked it.

Every drain kicks the consumer before it waits, which hid the bug for as
long as the raster thread drained whenever its queue ran dry.

**Rule.** In a store-then-load handshake (Dekker style), both stores and
both loads must be `SeqCst`. `Release`/`Acquire` orders a store with later
stores and a load with later loads, never a store with a later load.
