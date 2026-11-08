# The pinned MIPS guest compiler: GCC 16.2.1 for mips64-linux-gnu. Fedora 45 is
# the only distribution packaging GCC 16.2 as a MIPS cross compiler, and its
# release repository never changes, so these exact versions stay installable
# until Fedora 45 reaches end of life (2027-12-08); then point --repo at the
# archive or move to a newer release. Bumping the compiler changes every
# benchmark figure: rebuild bench/prebuilt in the same commit. The version
# below must match GUEST_GCC_VERSION in toolchain.mk.
# The base is a multi-architecture digest and runs natively on amd64 and arm64.
FROM docker.io/library/fedora:45@sha256:a7d72ba2745b100a3351087472e5ee7b2450af7cf331e7673007e3817612839c
RUN dnf -y --repo=fedora --setopt=install_weak_deps=False install \
        gcc-mips64-linux-gnu-16.2.1-1.fc45 \
        cross-gcc-common-16.2.1-1.fc45 \
        binutils-mips64-linux-gnu-2.47-1.fc45 make \
    && dnf clean all
