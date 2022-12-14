#!/bin/bash
cd "$(dirname "$0")"
numactl -C 0-3 pytest -xvs pyfiles
