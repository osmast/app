# DaphniaScope — convenience targets (conda env: z-app)
ENV ?= z-app
PY  := conda run -n $(ENV) python

#: The screen that ``make ui-designer`` opens when FILE is not given.
FILE ?= pages.ui

.PHONY: env env-update run db db-reset test ui-check ui-designer ui-list stream stream-publish mediamtx-setup mediamtx-run \
        train train-check train-inspect train-pose train-pose-inspect repair-boxes clean

env: ## create the conda environment
	conda env create -f environment.yml

env-update: ## sync the conda environment with environment.yml
	conda env update -f environment.yml --prune

run: ## launch the desktop application
	$(PY) run.py

db: ## (re)initialise the SQLite database
	$(PY) scripts/init_db.py

db-reset: ## delete the local database (recreated on the next launch)
	rm -f zoo.db

test: ## run the test suite
	$(PY) -m unittest discover -s tests -t . -v

ui-check: ## paint every surface offscreen, full screen and back (no display needed)
	$(PY) scripts/check_ui.py

ui-designer: ## open a screen in Qt's UI editor (make ui-designer FILE=bench.ui)
	$(PY) scripts/edit_ui.py $(FILE)

ui-list: ## report what every screen declares, and what a Designer edit would break
	$(PY) scripts/edit_ui.py --list

ui-render: ## render every surface offscreen to PNGs (review a layout change by looking)
	$(PY) scripts/render_ui.py

stream: ## probe the configured MediaMTX stream
	$(PY) scripts/check_stream.py

stream-publish: ## publish an ffmpeg test pattern to MediaMTX over SRT
	./scripts/dev_publish_stream.sh

mediamtx-setup: ## install config/mediamtx.yml system-wide
	./scripts/setup_mediamtx.sh

mediamtx-run: ## run MediaMTX with the repository config
	mediamtx config/mediamtx.yml

train-check: ## prove PyTorch + MPS + OpenCV work before a long run
	$(PY) scripts/check_torch.py

train-inspect: ## describe the dataset export the trainer would read
	$(PY) scripts/train_rtmdet.py --inspect

train: ## train RTMDet on datasets/bb (see docs/rtmdet.md for the options)
	$(PY) scripts/train_rtmdet.py --data datasets/bb --report models/rtmdet_report.json

train-pose: ## train RTMPose on datasets/kp (see docs/rtmpose.md for the options)
	$(PY) scripts/train_rtmpose.py --data datasets/kp --report models/rtmpose_report.json

train-pose-inspect: ## describe the keypoint export the pose trainer would read
	$(PY) scripts/train_rtmpose.py --inspect --data datasets/kp

repair-boxes: ## report bounding boxes that leave their image (dry run)
	$(PY) scripts/repair_boxes.py

clean: ## remove caches and the local database
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f zoo.db
