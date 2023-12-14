#
## Developer
### Install requirements
- set local python version (this will load required python version from .python-version file)
```Bash
pyenv install $(cat .python-version)
pyenv local
```
- create virtualenv
```Bash
python3 -m venv .venv
```
- activate virtualenv
```Bash
source .venv/bin/activate
```
- install poetry
```Bash
./scripts/install-poetry.sh
```
- install dependencies
```Bash
poetry install
```