# Stress test IRIX memory

`memstress.c` writes and verifies every 64-bit word in an anonymous mapping.
Each word includes its index and its complement. Four passes XOR these values
with zeroes, ones, alternating `0x55` bits, and alternating `0xaa` bits.
Verification alternates between ascending and descending addresses.

## Run the test

On an IRIX guest with MIPSpro installed, compile the 64-bit executable:

```sh
cc -64 -mips4 -O2 -o memstress memstress.c
```

On a guest configured with `[512, 512, 0, 0]`, run an 896 MB allocation:

```sh
./memstress 896 4
```

To check a complete 1024 MB allocation, run:

```sh
./memstress 1024 4
```

Use the 64-bit build for the 1024 MB test. The tested n32 executable cannot
obtain one contiguous mapping of that size.

Each pass must finish with `errors=0`. The program exits with status 1 on a
data mismatch and status 2 on an allocation or accounting error. A successful
run prints `PASS ALL` and exits with status 0.

## Check physical residency

The program attempts `mlock`; `locked=0` means the allocation is not pinned.
While it runs, use the PID from its `START` line to collect residency and swap
use through a second terminal:

```sh
ps -o pid,rss,sz,args -p PID
swap -s
swap -l
```

The test prints the page size and physical-page count returned by `sysmp`,
along with changes in the kernel's `bswapin` and `bswapout` counters. On the
tested IP28 guest, pages are 16384 bytes; multiply `RSS` by that size to obtain
resident bytes. Distinguish allocated virtual memory from resident memory.

IRIX's kernel and services also occupy RAM. A 1024 MB application allocation
cannot reside entirely in a machine containing exactly 1024 MB of physical
RAM. That run exercises paging too. Use an allocation that leaves room for
the operating system, confirm its residency, and confirm zero swap activity
to establish physical RAM coverage.
