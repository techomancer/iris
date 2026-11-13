#!/bin/sh

mkdir -p 6.5.22 6.5.22/ovl1 6.5.22/ovl2 6.5.22/ovl3 6.5.22/apps
mkdir -p 6.5.30 6.5.30/ovl1 6.5.30/ovl2 6.5.30/ovl3 6.5.30/apps
mkdir -p f1 f2 devl devf13 nfs mipspro7.4

if [ ! -d 6.5.22/ovl1/dist ] ; then
	if [ ! -f 'IRIX 6.5.22 Installation Tools and Overlays (1 of 3).tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5.22%20Installation%20Tools%20and%20Overlays%20%281%20of%203%29.tar
	else
		tar xf 'IRIX 6.5.22 Installation Tools and Overlays (1 of 3).tar' -C 6.5.22/ovl1
	fi
fi

if [ ! -d 6.5.22/ovl2/dist ] ; then
	if [ ! -f 'IRIX 6.5.22 Overlays (2 of 3).tar' ] ; then
		 echo Please download https://jrra.zone/sgi/tar/IRIX%206.5.22%20Overlays%20%282%20of%203%29.tar
	else
		tar xf 'IRIX 6.5.22 Overlays (2 of 3).tar' -C 6.5.22/ovl2
	fi
fi

if [ ! -d 6.5.22/ovl3/dist ] ; then
	if [ ! -f 'IRIX 6.5.22 Overlays (3 of 3).tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5.22%20Overlays%20%283%20of%203%29.tar
	else
		tar xf 'IRIX 6.5.22 Overlays (3 of 3).tar' -C 6.5.22/ovl3
	fi
fi

if [ ! -d 6.5.22/apps/dist ] ; then
	if [ ! -f 'IRIX 6.5 Applications November 2003.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5%20Applications%20November%202003.tar
	else
		tar xf 'IRIX 6.5 Applications November 2003.tar'  -C 6.5.22/apps
	fi
fi

if [ ! -d 6.5.30/ovl1/dist ] ; then
	if [ ! -f 'IRIX 6.5.30 Installation Tools and Overlays (1 of 3).tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5.30%20Installation%20Tools%20and%20Overlays%20%281%20of%203%29.tar
	else
		tar xf 'IRIX 6.5.30 Installation Tools and Overlays (1 of 3).tar' -C 6.5.30/ovl1
	fi
fi

if [ ! -d 6.5.30/ovl2/dist ] ; then
	if [ ! -f 'IRIX 6.5.30 Overlays (2 of 3).tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5.30%20Overlays%20%282%20of%203%29.tar
	else
		tar xf 'IRIX 6.5.30 Overlays (2 of 3).tar' -C 6.5.30/ovl2
	fi
fi

if [ ! -d 6.5.30/ovl3/dist ] ; then
	if [ ! -f 'IRIX 6.5.30 Overlays (3 of 3).tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5.30%20Overlays%20%283%20of%203%29.tar
	else
		tar xf 'IRIX 6.5.30 Overlays (3 of 3).tar' -C 6.5.30/ovl3
	fi
fi

if [ ! -d 6.5.30/apps/dist ] ; then
	if [ ! -f 'IRIX 6.5 Applications August 2006.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5%20Applications%20August%202006.tar
	else
		tar xf 'IRIX 6.5 Applications August 2006.tar' -C 6.5.30/apps
	fi
fi

if [ ! -d f1/dist ] ; then
	if [ ! -f 'IRIX 6.5 Foundation 1.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5%20Foundation%201.tar
	else
		tar xf 'IRIX 6.5 Foundation 1.tar' -C f1
	fi
fi

if [ ! -d f2/dist ] ; then
	if [ ! -f 'IRIX 6.5 Foundation 2.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5%20Foundation%202.tar
	else
		tar  xf 'IRIX 6.5 Foundation 2.tar' -C f2
	fi
fi

if [ ! -d nfs/dist ] ; then
	if [ ! -f 'ONC3 NFS Version 3.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/ONC3%20NFS%20Version%203.tar
	else
		tar xf 'ONC3 NFS Version 3.tar' -C nfs
	fi
fi

if [ ! -d devl/dist ] ; then
	if [ ! -f 'IRIX 6.5 Development Libraries February 2002.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%206.5%20Development%20Libraries%20February%202002.tar
	else
		tar xf 'IRIX 6.5 Development Libraries February 2002.tar' -C devl
	fi
fi

if [ ! -d devf13/dist ] ; then
	if [ ! -f 'IRIX Development Foundation 1.3.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/IRIX%20Development%20Foundation%201.3.tar
	else
		tar xf 'IRIX Development Foundation 1.3.tar' -C devf13
	fi
fi

if [ ! -d mipspro7.4/dist ] ; then
	if [ ! -f 'MIPSpro C Compiler 7.4.tar' ] ; then
		echo Please download https://jrra.zone/sgi/tar/MIPSpro%20C%20Compiler%207.4.tar
	else
		tar xf 'MIPSpro C Compiler 7.4.tar' -C  mipspro7.4
	fi
fi

