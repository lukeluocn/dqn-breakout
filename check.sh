#!/bin/bash
for script in $(ls *.py); do
    echo "On checking ${script}:"
    python -m mypy --ignore-missing-imports ${script}
done