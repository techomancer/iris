@echo off

if "%1%" == "clean" goto clean

if "%1%" == "" (
set DIR=.
echo Searching IRIX tar files in current directory
)
if not "%1%" == "" (
set DIR=%1%
echo Searching IRIX tar files in %DIR%
)

if not exist 6.5.22 mkdir 6.5.22
cd 6.5.22

if not exist ovl1 mkdir ovl1
if not exist ovl2 mkdir ovl2
if not exist ovl3 mkdir ovl3
if not exist apps mkdir apps
cd ovl1
if not exist dist (
if not exist "%DIR%\IRIX 6.5.22 Installation Tools and Overlays (1 of 3).tar" ( 
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5.22%%20Installation%%20Tools%%20and%%20Overlays%%20%%281%%20of%%203%%29.tar" 
) else tar -xf "%DIR%\IRIX 6.5.22 Installation Tools and Overlays (1 of 3).tar"
)
cd ..

cd ovl2
if not exist dist (
if not exist "%DIR%\IRIX 6.5.22 Overlays (2 of 3).tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5.22%%20Overlays%%20%%282%%20of%%203%%29.tar" 
) else tar -xf "%DIR%\IRIX 6.5.22 Overlays (2 of 3).tar"
) 
cd ..

cd ovl3
if not exist dist (
if not exist "%DIR%\IRIX 6.5.22 Overlays (3 of 3).tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5.22%%20Overlays%%20%%283%%20of%%203%%29.tar"
) else tar -xf "%DIR%\IRIX 6.5.22 Overlays (3 of 3).tar"
)
cd ..

cd apps
if not exist dist (
if not exist "%DIR%\IRIX 6.5 Applications November 2003.tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5%%20Applications%%20November%%202003.tar"
) else tar -xf "%DIR%\IRIX 6.5 Applications November 2003.tar"
)
cd ..

cd ..
if not exist 6.5.30 mkdir 6.5.30
cd 6.5.30


if not exist ovl1 mkdir ovl1
if not exist ovl2 mkdir ovl2
if not exist ovl3 mkdir ovl3
if not exist apps mkdir apps
cd ovl1
if not exist dist (
if not exist "%DIR%\IRIX 6.5.30 Installation Tools and Overlays (1 of 3).tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5.30%%20Installation%%20Tools%%20and%%20Overlays%%20%%281%%20of%%203%%29.tar"
) else tar -xf "%DIR%\IRIX 6.5.30 Installation Tools and Overlays (1 of 3).tar"
)
cd ..

cd ovl2
if not exist dist (
if not exist "%DIR%\IRIX 6.5.30 Overlays (2 of 3).tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5.30%%20Overlays%%20%%282%%20of%%203%%29.tar"
) else tar -xf "%DIR%\IRIX 6.5.30 Overlays (2 of 3).tar"
)
cd ..

cd ovl3
if not exist dist (
if not exist "%DIR%\IRIX 6.5.30 Overlays (3 of 3).tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5.30%%20Overlays%%20%%283%%20of%%203%%29.tar"
) else tar -xf "%DIR%\IRIX 6.5.30 Overlays (3 of 3).tar"
)
cd ..

cd apps
if not exist dist (
if not exist "%DIR%\IRIX 6.5 Applications August 2006.tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5%%20Applications%%20August%%202006.tar"
) else tar -xf "%DIR%\IRIX 6.5 Applications August 2006.tar"
)
cd ..


cd ..
if not exist f1 mkdir f1
if not exist f2 mkdir f2
if not exist nfs mkdir nfs

cd f1
if not exist dist (
if not exist "%DIR%\IRIX 6.5 Foundation 1.tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5%%20Foundation%%201.tar"
) else tar -xf "%DIR%\IRIX 6.5 Foundation 1.tar"
)
cd ..

cd f2
if not exist dist (
if not exist "%DIR%\IRIX 6.5 Foundation 2.tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5%%20Foundation%%202.tar"
) else tar -xf "%DIR%\IRIX 6.5 Foundation 2.tar"
)
cd ..

cd nfs
if not exist dist (
if not exist "%DIR%\ONC3 NFS Version 3.tar" (
echo Please download "https://jrra.zone/sgi/tar/ONC3%%20NFS%%20Version%%203.tar"
) else tar -xf "%DIR%\ONC3 NFS Version 3.tar"
)
cd ..

if not exist devl mkdir devl
if not exist devf13 mkdir devf13
cd devf13
if not exist dist (
if not exist "%DIR%\IRIX Development Foundation 1.3.tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%20Development%%20Foundation%%201.3.tar" 
) else tar -xf "%DIR%\IRIX Development Foundation 1.3.tar"
)
cd ..
cd devl
if not exist dist (
if not exist "%DIR%\IRIX 6.5 Development Libraries February 2002.tar" (
echo Please download "https://jrra.zone/sgi/tar/IRIX%%206.5%%20Development%%20Libraries%%20February%%202002.tar"
) else tar -xf "%DIR%\IRIX 6.5 Development Libraries February 2002.tar"
)
cd ..
if not exist mipspro7.4 mkdir mipspro7.4
cd mipspro7.4
if not exist dist (
if not exist "%DIR%\MIPSpro C Compiler 7.4.tar" (
echo Please download "https://jrra.zone/sgi/tar/MIPSpro%%20C%%20Compiler%%207.4.tar"
) else tar -xf "%DIR%\MIPSpro C Compiler 7.4.tar"
)
cd ..

goto end

:clean
rd /s /q 6.5.22 6.5.30 f1 f2 nfs devl devf13 mipspro7.4

:end