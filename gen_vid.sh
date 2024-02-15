#!/usr/bin/env bash
ffmpeg -framerate 30 -pattern_type glob -i 'img/*.png' -c:v libx264 -pix_fmt yuv420p output.mp4
