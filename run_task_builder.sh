#!/bin/bash
export LIBGL_ALWAYS_SOFTWARE=1
export QT_X11_NO_MITSHM=1
export QT_QPA_PLATFORM=xcb
export QT_XKB_CONFIG_ROOT=/usr/share/X11/xkb
export DISPLAY=:1
python tools/task_builder.py "$@"
