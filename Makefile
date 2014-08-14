.PHONY: docs release clean build html

clean:
	rm -rf ${HOME}/pambox_test_env htmlcov

build: clean
	export PATH=~/miniconda/bin:${PATH}
	conda create -p ${HOME}/pambox_test_env --yes --file requirements.txt pip \
	&& source activate pambox_test_env \
	&& python setup.py install

test: clean build
	export PATH=~/miniconda/bin:${PATH}
	source activate pambox_test_env \
	&& conda install --yes --file testing_requirements.txt \
    && coverage run --source=pambox setup.py test \
    && coverage html \
    && coverage report

docs: html
	cd docs; make html

release: test docs
	vim pambox/__init__.py
