## 
# FILE: Makefile
# AUTHOR: Simon Sedlacek
# EMAIL: xsedla1h@stud.fit.vutbr.cz

train:
	python src/main.py --mode=train --task=$(task)

replay:
	python src/main.py --mode=replay --agent_path=$(path)

install:
	pip install -r requirements.txt

pack:
	tar -zcvf xsedla1h.tar.gz src requirements.txt Makefile README.md eval_notebook.ipynb doc eval_cartpole.sh agents

clean:
	rm -rf src/__pycache__
