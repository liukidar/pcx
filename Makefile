venv_name = venv
venv_activate_path := ./$(venv_name)/bin/activate
package_name = pcax
cov_args := --cov $(package_name)

.PHONY: clean venv lint test slowtest cov slowcov docs

clean:
	rm -rf ./$(venv_name)

venv:
	python3 -m venv $(venv_name) ;\
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r requirements/requirements-dev.txt ;\
	pip install --upgrade -r requirements/requirements.txt

update:
	. $(venv_activate_path) ;\
	pip install --upgrade pip setuptools wheel ;\
	pip install --upgrade -r requirements/requirements-dev.txt ;\
	pip install --upgrade -r requirements/requirements.txt

lint:
	. $(venv_activate_path) ;\
	flake8 $(package_name)/ ;\
	flake8 tests/

format:
	. $(venv_activate_path) ;\
	isort -rc . ;\
	autoflake -r --in-place --remove-unused-variables $(package_name)/ ;\
	autoflake -r --in-place --remove-unused-variables tests/ ;\
	black $(package_name)/ ;\
	black tests/

checkformat:
	. $(venv_activate_path) ;\
	isort -rc . ;\
	black $(package_name)/ --check ;\
	black tests/ --check

test:
	. $(venv_activate_path) ;\
	py.test

cov:
	. $(venv_activate_path) ;\
	py.test $(cov_args)

