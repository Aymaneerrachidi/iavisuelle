.PHONY: install data dev test lint check docker

install:
	pip install -e ".[ml,dev,data]"

data:
	python data/download_flickr8k.py

dev:
	vlm-lab --reload

test:
	pytest --cov=vlm_lab --cov-report=term-missing

lint:
	ruff check .

check: lint test
	python -m compileall -q src

docker:
	docker build -t prism-vlm-lab .

