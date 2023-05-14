DEFAULT_VENV_DIR=.venv

.PHONY: all py3.10

all: data py3.10

py3.10: $(DEFAULT_VENV_DIR)/py3.10/

$(DEFAULT_VENV_DIR)/py%/:
	python3.10 -m venv $@
	$@/bin/pip install -U pip
	$@/bin/pip install -r pip-requirements.txt
	$@/bin/pip install -r pip-requirements-dev.txt
	$@/bin/python3 -m ipykernel install --user --name=pred-ops-os
	@echo "Run \`source $@/bin/activate\` to start the virtual env."

data:
	wget https://zenodo.org/record/7934656/files/INTOPS2023-data.tar.bz2
	tar --use-compress-program=pbzip2 -xvf INTOPS2023-data.tar.bz2
