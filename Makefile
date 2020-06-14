build:
	docker build -t spacy-fasttext .

train:
	docker run --name fasttext -v $(PWD):/app spacy-fasttext python load_fastText.py

package:
	# docker run -v $(PWD):/app spacy-fasttext python -m spacy package -m pt_model model model_package
	docker run -it -v $(PWD):/app spacy-fasttext 