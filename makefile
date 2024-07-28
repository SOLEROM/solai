PHONY: all help build deploy

# Default target
all: help

labs:
	_jupBook/labs_md_script.sh	00_ai24

# Help target
help:
	@echo "Makefile for building and deploying the Jupyter book"
	@echo "output will be at https://solerom.github.io/solai"
	@echo
	@echo "Usage:"
	@echo "  make build      Build the Jupyter book"
	@echo "  make deploy     Deploy the Jupyter book to GitHub Pages"
	@echo "  make help       Display this help message"
	@echo "  make local      show local version"
	@echo "  make remote      show remote version"


# Build target
build: labs
	rm -rf _build
	jupyter-book build . --toc ./_toc.yml --config _jupBook/_config.yml
	@firefox _build/html/index.html

local:
	@firefox _build/html/index.html

remote: 
	@firefox https://solerom.github.io/solai

# Deploy target
deploy: build
	## using the ghp-import will create and deploy the book to new gh-pages branch that is rendered as github pages;
	ghp-import -n -p -f _build/html
	@echo "after action see https://solerom.github.io/solai"

