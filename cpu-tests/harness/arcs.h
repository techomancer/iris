/* arcs.h — console output through the PROM's ARCS firmware vector.
 *
 * The harness drives SCC channel B directly, which is invisible on a machine
 * whose PROM console is graphics. ARCS Write() goes wherever the PROM's own
 * console goes, so the suite becomes readable on screen — and it costs nothing
 * when no PROM is present, because the probe simply fails.
 */
#ifndef ARCS_H
#define ARCS_H

#include "console.h"

int  arcs_probe(void);          /* nonzero if a usable firmware vector was found */
void arcs_putc(int c);          /* line-buffered; no-op until arcs_probe succeeds */
void arcs_flush(void);

/* Write through the PROM's disk driver; see arcs.c for why not scsilog's. */
int  arcs_disk_write(const char *path, u64 offset, const void *buf, u32 len);

#endif /* ARCS_H */
