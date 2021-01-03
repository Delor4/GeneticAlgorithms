CMDS = help build run install

.PHONY: $(CMDS)

help:
	@echo Possible targets: $(CMDS)

all: run

build:
	python setup.py build

run:
	python linear_equations.py

install:
    pip install requirements.txt
