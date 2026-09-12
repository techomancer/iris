# Prediction for the next Indy R4400 run

> **Outcome, recorded after the fact:** 16 failures → 3, not the 1 predicted.
> `cache/index_tag_rt` passed (the hazard theory was right),
> `cp0/random_respects_wired` did not (the warm-up theory was wrong),
> and `fpu/cmp_snan_any_pred` was over-corrected. Kept as written because a
> prediction is only worth anything if it is left alone afterwards.

Written **before** the run, against ELF `bcc74600e5b96ac9a6cb0b9de2b19a86`, so
the result can falsify it rather than be rationalised after the fact.

Baseline this replaces: the first hardware run (ELF `5d57556e`) reported
`2097 checks passed, 64 failed (240 tests)` — 16 tests failing.

## Expected outcome

**Exactly one failing test: `fpu/vec_cvt_from_l`.** Everything else passes.

If that holds, all 15 corrected tests now agree with silicon, and the remaining
one is the only open question.

### Corrected from hardware evidence — expect PASS (13)

| test | what changed |
|---|---|
| `mips4/fp_cond_move_s` | COP1 unimplemented raises `EXC_FPE`, not `EXC_RI` |
| `mips4/fp_cond_move_d` | same |
| `mips4/recip_rsqrt` | same |
| `mips4/recip_rsqrt_d` | same |
| `identity/prid` | compare the implementation field only; rev 6.0 is not rev 4.0 |
| `mem/lwr_all_offsets` | a partial `LWR` preserves the upper half; only the 4-byte load sign-extends |
| `excep/cop2_unusable` | with `CU2` set the R4400 takes no exception at all |
| `fpu/qnan_operand` | quiet NaN propagates as `0x7FBFFFFF`, no trap |
| `fpu/snan_operands` | signalling NaN raises Unimplemented Operation, sets no flag |
| `fpu/compare_nan` | a quiet NaN compare raises Invalid |
| `fpu/cmp_signalling_qnan` | …on every predicate, not only the signalling half |
| `fpu/cmp_snan_any_pred` | a signalling NaN compare raises none |
| `fpu/cmp_trap_on_signal` | with Invalid enabled even a non-signalling predicate traps |

### Reasoned, not proven — expect PASS but these are the ones to watch (2)

| test | hypothesis |
|---|---|
| `cache/index_tag_rt` | `CACHE_OP` had no hazard spacing, so `TagLo` was read before `Index_Load_Tag` landed. Added `CACHE_TAG_HAZARD()`. If it still fails, the spacing is insufficient or `Index_Store_Tag` is not taking. |
| `cp0/random_respects_wired` | writing `Wired` does not reload `Random`; it keeps counting down until it wraps. Added a 512-sample warm-up. If it still fails, note the new `low` count — if it is still ~250 the warm-up is not the mechanism. |

### Deliberately untouched — expect FAIL (1)

`fpu/vec_cvt_from_l`. `cvt.s.l` returned `0x1000` with no flags where a
converted value was expected, and `cvt.d.l` returned an unrelated negative
double. One run is not enough to tell an Unimplemented trap from a decode
difference, and guessing would be exactly the "record what it did and call it
expected" trap `PLAN.md` §6 warns about. Needs its own investigation.

## Regression check

The 15 tests that fail under IRIS and passed on hardware were **not modified**.
They must still pass. If any of them now fails, a change here broke something.

## How to verify

```sh
run/extract-log.py <card>/HD20*.img -o cputest-hw-r4400-v2.log
run/diff-hw.py build/emu-r4400.log cputest-hw-r4400-v2.log
```

`IRIS-WRONG` should contain `fpu/vec_cvt_from_l` and nothing else. `IRIS-BUG`
should be the same 15 as before, plus the 16 newly-corrected tests that IRIS
gets wrong — 31 in total, which is the emulator's real bug surface.
