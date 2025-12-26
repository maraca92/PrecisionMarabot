#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
python -c "import aiofiles; print('aiofiles installed successfully')"
