# App Nap throttles the Mac app's background work unless it holds an activity

**Symptom.** Converting a 4.3 GB CHD to zlib took about 390 s in the
Mac app with it in the background, against 86–116 s for the same work
from a command-line tool. A disk sync after a stop was not slowed.

**Cause.** macOS App Nap lowers the CPU and I/O priority of an app that is
not in front and shows no sign of working. A running machine usually has its
audio output open, which keeps the app awake, so the syncs that follow a stop
ran at full speed; a conversion with no machine running had nothing.

**Fix.** The AppKit shell holds
`NSProcessInfo.beginActivityWithOptions(UserInitiatedAllowingIdleSystemSleep)`
while a machine runs, disks synchronize, a disk converts or the disk browser
writes (`keep_awake` in `frontends/appkit/src/app.rs`), and ends it when the
work stops. The same conversion then took 95 s. The option still lets the
Mac sleep when idle.

**Rule.** Any new long-running job in the Mac app must count in `keep_awake`.
A command-line timing of the same work is the reference to compare against.
