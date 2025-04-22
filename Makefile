HF_REPO=https://$(HF_USERNAME):$(HF_TOKEN)@huggingface.co/spaces/$(HF_USERNAME)/$(HF_SPACE_NAME)

install:
	pip install -r requirements.txt

format:
	python -m black *.py

train:
	python train.py  # this should generate `Results/`

eval:
	@echo "Evaluation done in train.py or optional step"

hf-login:
	git pull origin main
	git switch main
	pip install -U "huggingface_hub[cli]"
	@if [ -z "$(HF_TOKEN)" ]; then echo "Error: HF token is missing!"; exit 1; fi
	huggingface-cli login --token $(HF_TOKEN) --add-to-git-credential

push-hub:
	@if [ -z "$(HF_TOKEN)" ]; then echo "Error: HF token is missing!"; exit 1; fi
	huggingface-cli upload Sreenidhi31/Enhanced-OSCC ./App --repo-type=space --commit-message "Sync App files"

deploy: hf-login push-hub
