SHELL := bash
PYTHON_NAME = rhasspyserver_hermes
PACKAGE_NAME = rhasspy-server-hermes
SOURCE = $(PYTHON_NAME)
PYTHON_FILES = $(SOURCE)/*.py *.py
SHELL_FILES = bin/* debian/bin/*

.PHONY: reformat check venv dist sdist pyinstaller debian docker deploy

version := $(shell cat VERSION)
architecture := $(shell bash architecture.sh)

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

reformat:
	scripts/format-code.sh $(PYTHON_FILES)

check:
	scripts/check-code.sh $(PYTHON_FILES)

venv: rhasspy-libs
	scripts/create-venv.sh

dist: sdist debian

sdist:
	python3 setup.py sdist

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------

docker: requirements_rhasspy.txt
	docker build . -t "rhasspy/$(PACKAGE_NAME):$(version)" -t "rhasspy/$(PACKAGE_NAME):latest"

deploy:
	echo "$$DOCKER_PASSWORD" | docker login -u "$$DOCKER_USERNAME" --password-stdin
	docker push "rhasspy/$(PACKAGE_NAME):$(version)"

# -----------------------------------------------------------------------------
# Debian
# -----------------------------------------------------------------------------

pyinstaller:
	scripts/build-pyinstaller.sh "${architecture}" "${version}"

debian:
	scripts/build-debian.sh "${architecture}" "${version}"

# -----------------------------------------------------------------------------
# Downloads
# -----------------------------------------------------------------------------

requirements_rhasspy.txt: requirements.txt
	grep '^rhasspy-' $< | sed -e 's|=.\+|/archive/master.tar.gz|' | sed 's|^|https://github.com/rhasspy/|' > $@

# Rhasspy development dependencies
rhasspy-libs: $(DOWNLOAD_DIR)/rhasspy-profile-0.1.3.tar.gz $(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz $(DOWNLOAD_DIR)/rhasspy-nlu-0.1.6.tar.gz $(DOWNLOAD_DIR)/rhasspy-supervisor-0.1.3.tar.gz

$(DOWNLOAD_DIR)/rhasspy-profile-0.1.3.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-profile/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-hermes-0.1.6.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-profile/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-nlu-0.1.6.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-profile/archive/master.tar.gz"

$(DOWNLOAD_DIR)/rhasspy-supervisor-0.1.3.tar.gz:
	mkdir -p "$(DOWNLOAD_DIR)"
	curl -sSfL -o $@ "https://github.com/rhasspy/rhasspy-profile/archive/master.tar.gz"
