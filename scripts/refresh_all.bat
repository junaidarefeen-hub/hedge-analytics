@echo off
REM Daily cache refresh for hedge-analytics (factor data + market monitor).
REM Scheduled via Windows Task Scheduler: weekdays at 4:45 PM ET.

cd /d C:\Users\jarefeen\hedge-analytics
C:\local\python\miniforge3\python.exe scripts\refresh_all.py --commit >> scripts\refresh_all.log 2>&1
