
train:
	python src/main.py --mode=train --task=$(task)

replay:
	python src/main.py --mode=replay --agent_path=$(path)
