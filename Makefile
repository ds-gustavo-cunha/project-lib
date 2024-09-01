install_project_lib:
	pip install -e .

install_requirements:
	@pipenv install -r requirements.txt

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="tests/*.py"

check_code:
	black project_lib/*.py scripts/*