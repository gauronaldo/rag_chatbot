.PHONY: check install test clean start pull_model tidy eval-w18347 eval-w18347-fast

check:
	which pip3
	which python3

install:
	echo "Installing..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

pull_model:
	ollama pull qwen2.5:7b

start:
	streamlit run streamlit_app.py

tidy:
	ruff format --exclude=.venv .
	ruff check --exclude=.venv . --fix

test:
	pytest -q tests/test_rag_mvp.py

eval-w18347:
	python -m rag_mvp.run_evaluation --w18347-all

eval-w18347-fast:
	python -m rag_mvp.run_evaluation --w18347-all --skip-ragas

check-formatting:
	ruff format . --check

clean:
	echo "Cleaning virtual environment..."
	rm -rf .venv
	echo "Cleaning all compiled Python files..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	echo "Cleaning the cache..."
	rm -rf .pytest_cache
	rm -rf .ruff_cache
