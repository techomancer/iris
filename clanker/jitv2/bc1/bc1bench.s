/* bc1loop(n) — a hand-written hot loop whose only interesting content is an
 * FP compare followed by BC1T. Written in assembly on purpose: every C
 * version of this got folded or hoisted by MIPSpro -Ofast, and the whole
 * point is to control the instruction mix exactly.
 *
 * Each iteration executes: c.lt.d, bc1t, its delay slot, one addiu on
 * whichever arm runs, and the loop's own bgtz. The BC1 alternates taken and
 * not-taken (the compare operands swap every iteration) so it cannot settle
 * into one trivially-predicted direction.
 *
 * n32 ABI: n arrives in $4, result returns in $2.
 */
	.rdata
	.align	3
dconst:
	.double	1.0
	.double	2.0

	.text
	.align	2
	.globl	bc1loop
	.ent	bc1loop
bc1loop:
	.set	noreorder
	.set	noat
	la	$8, dconst
	ldc1	$f0, 0($8)		/* $f0 = 1.0 */
	ldc1	$f2, 8($8)		/* $f2 = 2.0 */
	move	$2, $0			/* accumulator */

loop:
	c.lt.d	$f0, $f2		/* cc0 = (f0 < f2) */
	bc1t	taken
	nop
	addiu	$2, $2, 1		/* not-taken arm */
	b	next
	nop
taken:
	addiu	$2, $2, 2		/* taken arm */
next:
	/* swap $f0/$f2 so the next compare goes the other way */
	mov.d	$f4, $f0
	mov.d	$f0, $f2
	mov.d	$f2, $f4

	addiu	$4, $4, -1
	bgtz	$4, loop
	nop

	jr	$31
	nop
	.set	at
	.set	reorder
	.end	bc1loop
