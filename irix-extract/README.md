# Installation guide for IRIX 6.5.22 / 6.5.30 using IRIS built-in TFTP and RSH service


## Extract IRIX CDROM images

Download the .tar version IRIX CDROM images and extract to the folder list in the table bellow.
You can use ```extract.cmd``` with argument of the directory absolute path storing the tar files:
```
extract.cmd c:\Users\xxx\Downloads
``` 
When a tar file is not found batch file will print the URL to download it.

Similar shell script is available for Linux and MacOS: ```extract.sh```. It works only from current directory.


The following images to be downloaded:

| CDROM | Directory | Installation source |
| ----- | --------- | ------------------- |
| [IRIX 6.5.22 Installation Tools and Overlays (1 of 3)](https://jrra.zone/sgi/tar/IRIX%206.5.22%20Installation%20Tools%20and%20Overlays%20%281%20of%203%29.tar) | 6.5.22/ovl1 | 192.168.0.1:6.5.22/ovl1/dist |
| [IRIX 6.5.22 Overlays (2 of 3)](https://jrra.zone/sgi/tar/IRIX%206.5.22%20Overlays%20%282%20of%203%29.tar) | 6.5.22/ovl2 | 192.168.0.1:6.5.22/ovl2/dist |
| [IRIX 6.5.22 Overlays (3 of 3)](https://jrra.zone/sgi/tar/IRIX%206.5.22%20Overlays%20%283%20of%203%29.tar) | 6.5.22/ovl3 | 192.168.0.1:6.5.22/ovl3/dist |
| [IRIX 6.5 Applications November 2003](https://jrra.zone/sgi/tar/IRIX%206.5%20Applications%20November%202003.tar) | 6.5.22/apps | 192.168.0.1:6.5.22/apps/dist |
| [IRIX 6.5.30 Installation Tools and Overlays (1 of 3)](https://jrra.zone/sgi/tar/IRIX%206.5.30%20Installation%20Tools%20and%20Overlays%20%281%20of%203%29.tar) | 6.5.30/ovl1 | 192.168.0.1:6.5.30/ovl1/dist |
| [IRIX 6.5.30 Overlays (2 of 3)](https://jrra.zone/sgi/tar/IRIX%206.5.30%20Overlays%20%282%20of%203%29.tar) | 6.5.30/ovl2 | 192.168.0.1:6.5.30/ovl2/dist |
| [IRIX 6.5.30 Overlays (3 of 3)](https://jrra.zone/sgi/tar/IRIX%206.5.30%20Overlays%20%283%20of%203%29.tar) | 6.5.30/ovl3 | 192.168.0.1:6.5.30/ovl3/dist |
| [IRIX 6.5 Applications August 2006](https://jrra.zone/sgi/tar/IRIX%206.5%20Applications%20August%202006.tar) | 6.5.30/apps | 192.168.0.1:6.5.30/apps/dist |
| [IRIX 6.5 Foundation 1](https://jrra.zone/sgi/tar/IRIX%206.5%20Foundation%201.tar) | f1 | 192.168.0.1:f1/dist |
| [IRIX 6.5 Foundation 2](https://jrra.zone/sgi/tar/IRIX%206.5%20Foundation%202.tar) | f2 | 192.168.0.1:f2/dist |
| [ONC3/NFS Version 3 for IRIX 6.2, 6.3, 6.4 and 6.5](https://jrra.zone/sgi/tar/ONC3%20NFS%20Version%203.tar) | nfs | 192.168.0.1:nfs/dist/dist6.5 |
| [IRIX 6.5 Development Libraries](https://jrra.zone/sgi/tar/IRIX%206.5%20Development%20Libraries%20February%202002.tar) | devl | 192.168.0.1:devl/dist |
| [IRIX Development Foundation 1.3](https://jrra.zone/sgi/tar/IRIX%20Development%20Foundation%201.3.tar) | devf13 | 192.168.0.1:devf13/dist |
| [MIPSpro C Compiler 7.4](https://jrra.zone/sgi/tar/MIPSpro%20C%20Compiler%207.4.tar) | mipspro7.4 | 192.168.0.1:mipspro7.4/dist |

## IRIS configuration

Point tftp-dir (cli option ```--tftp-dir``` or toml setting ```network.tftp_dir```) 
and rsh-dir (cli option `--rsh-dir` or toml setting `network.rsh_dir`) to the folder containing extracted IRIX ISO images

```
[network]
mode           = "nat"
tftp_dir       = 'irix-extract'
rsh_dir	       = 'irix-extract'

```

## Partitioning

Boot the partitioning utility using the following command on 32-bit machines:

```boot -f bootp()192.168.0.1:6.5.22/ovl1/stand/fx.ARCS```

On 64-bit machines use

```boot -f bootp()192.168.0.1:6.5.22/ovl1/stand/fx.64```

```
Obtaining 6.5.22/ovl1/stand/fx.ARCS from server 192.168.0.1
95040+26448+7168+2805248+50056d+5908+8928 entry: 0x97d4aa40
SGI Version 6.5 ARCS BE  Oct  6, 2003
fx: "device-name" = (dksc)
fx: ctlr# = (0)
fx: drive# = (1)
...opening dksc(0,1,0)
dks0d1s10: volume header not valid
...drive selftest...OK
fx: Warning:  invalid label from disk driver, ignored
Scsi drive type == SGI     IRIS EMUL DISK  1.0
...creating default bootinfo
...created default partitions, use /repartition menu to change
...creating default volume directory

----- please choose one (? for help, .. to quit this menu)-----
[exi]t             [d]ebug/           [l]abel/           [a]uto
[b]adblock/        [exe]rcise/        [r]epartition/
fx> l

----- please choose one (? for help, .. to quit this menu)-----
[sh]ow/         [sy]nc          [se]t/          [c]reate/
fx/label> c

----- please choose one (? for help, .. to quit this menu)-----
[p]artitions   [s]giinfo      [b]ootinfo     [d]irectory    [a]ll
fx/label/create> a

...creating default bootinfo
...created default partitions, use /repartition menu to change
...creating default sgiinfo
...creating default volume directory

----- please choose one (? for help, .. to quit this menu)-----
[p]artitions   [s]giinfo      [b]ootinfo     [d]irectory    [a]ll
fx/label/create> ..

----- please choose one (? for help, .. to quit this menu)-----
[sh]ow/         [sy]nc          [se]t/          [c]reate/
fx/label> sy

writing label info to dksc(0,1,0)

----- please choose one (? for help, .. to quit this menu)-----
[sh]ow/         [sy]nc          [se]t/          [c]reate/
fx/label> ..

----- please choose one (? for help, .. to quit this menu)-----
[exi]t             [d]ebug/           [l]abel/           [a]uto
[b]adblock/        [exe]rcise/        [r]epartition/
fx> exi

```

## Installation

Boot from ```192.168.0.1``` remote directory ```6.5.22/ovl1/dist```

### Formating

Create filesystem.

### Source selection

Select extracted CDROM folders as installation source by loading files 6.5.22.txt or 6.5.30.txt

```Inst> admin load 192.168.0.1:6.5.22.txt```

Than proceed with the installation as done with CDROM image but you do not need to swap disks.

### Resolve conflicts

To resolve conflict on IRIX 6.5.22 load the conflicts.txt from RSH service or do it manually:

```Inst> admin source 192.168.0.1:conflicts.txt```

Than start the installation with command ```go```
