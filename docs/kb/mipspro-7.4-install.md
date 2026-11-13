# Installing MIPSpro 7.4 and 7.4.4m on IRIX 6.5

This guide documents the package order and `inst` selections needed for a
complete MIPSpro 7.4 C and C++ installation on IRIX 6.5. The procedure is
guest-side IRIX administration and does not depend on a particular emulator.

The procedure was tested with:

- iris `v2026-10-13-15-59`;
- an Indy IP24/R5000 guest;
- IRIX 6.5.8m;
- the N32 ABI and MIPS III ISA; and
- MIPSpro 7.4 followed by the 7.4.4m maintenance update.

It has not been verified on physical SGI hardware or other IRIX 6.5 maintenance
releases.

Use lawfully obtained SGI distribution media and any licenses required for
your environment. This guide does not provide compiler media, license material,
preinstalled images, or proprietary package contents.

## Before installing

Back up the system disk or take a known-good snapshot. Make all distributions
available inside IRIX, then run SGI's `inst` in the guest so that the
`/var/inst` product database remains consistent. Do not inject compiler files
directly into an IRIX filesystem from the host.

The required distributions are:

1. Compiler Execution Environment 7.4
2. IRIX Development Foundation 1.3
3. MIPSpro C 7.4
4. MIPSpro C++ 7.4
5. MIPSpro 7.4.4m update

Install them in that order. The 7.4.4m update overlays the base C and C++
products and must be applied last.

Users may delegate this setup to an automation system or an AI agent. An
automated operator should preserve the order above, stop on a nonzero `inst`
or compiler result, retain the command output, and use the executable probes
below as the acceptance test. Do not infer success from `cc -version` or from
the presence or absence of a particular diagnostic.

## Install the execution environment and development foundation

These products are normally present in an IRIX 6.5 development installation.
For each distribution, start `inst` with its `dist` directory:

```text
inst -f /path/to/distribution/dist
```

At the `inst` prompt, select the upgrades:

```text
keep *
install Upgrade
go
quit
```

Run this sequence first for Compiler Execution Environment 7.4, then for IRIX
Development Foundation 1.3. Review the selection before `go`; if either
product is absent rather than installed at an older revision, `Upgrade` may
select nothing and the installation must be adjusted for that system.

## Install the base C and C++ products

MIPSpro C and C++ may be absent even when `cc -version` reports an older
MIPSpro driver. In that case, `install Upgrade` selects nothing and does not
install the compiler frontends.

Start `inst` for the MIPSpro C 7.4 distribution and use:

```text
keep *
install all
go
quit
```

Repeat the same sequence for the MIPSpro C++ 7.4 distribution.

After these two installations, the expected frontend paths are:

```text
/usr/lib32/cmplrs/fec
/usr/lib32/cmplrs/fecc
```

`fec` is installed by the C product and `fecc` by the C++ product. A driver
version string alone is not evidence that either frontend is installed.

## Apply the 7.4.4m update

Start `inst` with the 7.4.4m update distribution only after both base compiler
products are installed. Use the tested selection sequence:

```text
keep all
install Upgrade
keep incompleteoverlays
go
quit
```

Check the `inst` result for errors before continuing.

## Compile and run C and C++ probes

Run the following from the Bourne shell. The C probe deliberately declares a
variable after a statement so that it also verifies the `-c99` mode needed by
some IRIX software.

```sh
set -eu

work=/var/tmp/mipspro-74-probe
rm -rf "$work"
mkdir "$work"
cd "$work"

cat > probe.c <<'EOF'
#include <stdio.h>

int main(void)
{
    puts("MIPSpro C probe passed");
    int result = 0;
    return result;
}
EOF

cat > probe.C <<'EOF'
#include <iostream>

int main()
{
    std::cout << "MIPSpro C++ probe passed" << std::endl;
    return 0;
}
EOF

cc -n32 -mips3 -O0 -c99 probe.c -o probe-c
./probe-c

CC -n32 -mips3 -O0 probe.C -o probe-cxx
./probe-cxx
```

Success requires all four commands to return zero and the two executables to
print:

```text
MIPSpro C probe passed
MIPSpro C++ probe passed
```

Reboot the system, confirm that both frontend paths still exist, and repeat
the probes if the installation is intended to become a reusable base image.

## License diagnostic

In the tested environment, both compiler drivers printed:

```text
No such feature exists (-5,116)
```

followed by the MIPSpro license-contact paragraph. With the complete stack
installed, the compilers nevertheless returned zero, produced the requested
executables, and those executables ran successfully.

Treat this only as an observed behavior of the tested environment, not as a
statement about licensing requirements or behavior on every installation. Do
not suppress or filter the message when determining readiness. Check the
compiler exit status, confirm that the output file was created, and execute
it. A nonzero status, a missing output file, or a failed program is an
installation failure even if the remaining output looks familiar.

## Troubleshooting

| Symptom | Check |
|---|---|
| `install Upgrade` selects no C or C++ packages | The product is probably absent. Install the base product with `install all`. |
| `cc` reports that it cannot execute `/usr/lib32/cmplrs/fec` | Confirm that MIPSpro C 7.4 was installed and that `fec` exists. |
| `CC` cannot execute `/usr/lib32/cmplrs/fecc` | Confirm that MIPSpro C++ 7.4 was installed and that `fecc` exists. |
| `cc` rejects `-c99` | Check that the 7.4 base products and the 7.4.4m update were applied in order. |
| A license paragraph is printed | Use the probe exit statuses and executed results; do not classify the installation from that text alone. |

