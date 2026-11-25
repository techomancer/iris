#!/bin/sh
# Bring the X session back after a test killed the server: re-arm
# autologin, make xdm restart the server, wait for the desktop.
touch /etc/autologin.on
pid=`ps -ef | grep -v grep | grep '/usr/bin/X11/xdm$' | awk '{print $2}' | head -1`
killall Xsgi 2>/dev/null
sleep 3
ps -ef | grep -v grep | grep -q Xsgi || kill -HUP $pid
n=0
while [ $n -lt 40 ] && ! ps -ef | grep -v grep | grep -q toolchest; do sleep 5; n=`expr $n + 1`; done
sleep 15
DISPLAY=:0 /usr/bin/X11/xset s reset 2>/dev/null
DISPLAY=:0 /usr/bin/X11/xset s off 2>/dev/null
