@echo off
for /d /r %%i in (__pycache__) do (
  echo Deleting folder: %%i
  rmdir /s /q "%%i"
)
echo All __pycache__ folders deleted.
