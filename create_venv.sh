#!/usr/bin/env bash

VENVNAME=sickern
python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip
test -f requirements.txt && pip install -r requirements.txt
deactivate