HF_REPO=https://$(HF_USERNAME):$(HF_TOKEN)@huggingface.co/spaces/$(HF_USERNAME)/$(HF_SPACE_NAME)

install:
	pip install -r requirements.txt

format:
	python -m black *.py

train:
	python train.py  # this should generate `Results/`

eval:
	@echo "Evaluation done in train.py or optional step"

update-branch:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add .
	git diff --cached --quiet || git commit -m "Auto-update: Results"
	git push || echo "No changes to commit"

deploy:
	rm -rf hf_space
	git clone $(HF_REPO) hf_space
	cp -r App hf_space/
	cp requirements.txt hf_space/
	cp README.md hf_space/ || true
	cp -r Data hf_space/ || true
	cp -r Model hf_space/ || true
	cp -r Results hf_space/ || true
	(cd hf_space && git add . && git commit -m "Auto-deploy from GitHub Actions" && git push)
	rm -rf hf_space
