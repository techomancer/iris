# ERAM transfers come once per GE11 (Maximum Impact)

Symptom: a GL app on Maximum Impact comes back from being iconified (or
from any context eviction) with its draw state wrong. Avid Matador
redrew into the wrong buffer, very dark, with stray lines. In the
`mgras trace` the batch-start registers come out shifted by one:
texmode1's value lands in txenv_rg, tram_cntrl's in max_pixel_rg, and
the stencil, colour-mask and PP1 values are junk.

Cause: when no ERAM slot is free, the kernel evicts a context with the
pass-through tokens, once per GE11 and in GE order, with no select token:

- save: `0xFC (S, 0x17C8)`, then `0xFC (S - 1, 0x17C8)`, each feeding a
  0x17C7-word DMA into consecutive halves of one host buffer. GE1's read
  output evidently runs a word late.
- restore: half 0 to S in 900-word `0xFD`/`0xFB` chunks, then `0xFA`;
  half 1 to S, then `0xFA`.

With one ERAM modelled, half 1 (read from S - 1) overwrote half 0 one
word off. That half holds our `Gl` struct.

Fix (`Hq3Engine::eram_pass_through`): a transfer repeating GE0's last one
is GE1's. That means a read at one below GE0's last address with the
same count, or a write set starting where GE0's last set started. GE1's
goes to `eram.words[1]`, its reads offset by one word. The HLE reads only
`words[0]`. Test: `gl_context_survives_two_ge_eviction`.
