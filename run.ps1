$Process = Start-Process "py" -ArgumentList "lstmtimeseries.py" -PassThru
wmic process where name="python" CALL setpriority "high priority"

#Set-ProcessPriority -ProcessId $Process.id -Priority BelowNormal