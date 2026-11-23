# The hot lane must work while the compile queue is full

**Keywords:** jitv2,compile-pool,hot-lane,promotion,queue-full,warm-up

A one-CPU Octane has one compile worker, busy for the whole IRIX boot
(3,400-3,800 compiles at 33-48 ms). The FIFO (`COMPILE_QUEUE_CAPACITY`, 2048)
sits full after login, so a program started then waits behind requests made
tens of seconds earlier. The hot lane exists to let it go first. It did not:

- A dropped FIFO push cleared the page's arrival count (`clear_scheduled`), so
  a page could not be promoted while the FIFO was full, and every arrival
  retried the FIFO: 50-150 million failed pushes per boot on the CPU thread.
  A dropped push now keeps the count (`drop_request`) and the arrival that
  makes the page hot goes to the hot lane.
- With 256 slots the lane filled with pages promoted a backlog ago, since
  every page waiting long enough reaches `HOT_PROMOTE_ARRIVALS`. It holds 8.
- A promotion the full lane refused left `promoted` set until the compile.
  The page now re-arms itself (`rearm_promotion`).

Measured, 100 repetitions of a delay loop built with `cc` after login, one
boot at a time on a loaded host: upstream 3.7-12.3 s interpreted (7 boots);
lane 256 with only the first change 10.5-12.8 s; lanes 8, 16, 32 with all
three 0.1-0.4, 0.7-1.5 and 1.2 s (3 boots each). Boot to login fell from
121-167 s to 101-122 s with fewer compiles.

Not the lever, measured the same way: four workers (`[jitv2] threads = 4`)
drained the queue during boot but left the warm-up at 3.7-4.4 s and more
than doubled the compiles, mostly whole-page recompiles; a page-level arrival
threshold of 4, 16 or 64 removed almost no compiles.
