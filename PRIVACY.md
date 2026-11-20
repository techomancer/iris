# Privacy Policy

_Last updated: November 15, 2026_

IRIS is an emulator of SGI workstations and servers, such as the Indy, Indigo2, O2 and Octane. It does **not** collect, store, transmit, or share any personal information.

**No data collection.** IRIS contains no analytics, telemetry, advertising, or tracking. It creates no accounts and sends no information about you or your usage to the developer or any third party.

**Camera.** If you enable the IndyCam (Video-In) feature, IRIS uses your computer's camera only to feed a live signal to the emulated Indy video-input device. The video is processed on your device and is never recorded, stored, or transmitted by IRIS.

**Networking.** IRIS does not contact any server of its own. Its emulated Ethernet is handled by an on-device userspace NAT; any network traffic originates from the guest operating system you choose to run and connects only where you direct it. A few features you turn on yourself also make connections you direct: a CD-ROM image given as an `http://` or `https://` address is read from that server, snapshots can be sent to or fetched from a server you name, and the optional VNC server accepts viewers on the address you choose.

**Your files.** Disk images, PROM files, and other files you open stay on your device. IRIS accesses only the files you explicitly select and uploads nothing unless you send a snapshot to a server you name.

**Children's privacy.** IRIS collects no data from anyone, including children.

**Changes.** Any updates to this policy will be posted on this page.

**Contact.** Questions about this policy: https://github.com/techomancer/iris/issues
