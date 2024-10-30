BASE_IMAGE = ghcr.io/allenai/pytorch:2.5.1-cuda12.1-python3.11-v2024.10.29

# NOTE: when upgrading the nightly version you also need to upgrade the torch version specification
# in 'pyproject.toml' to include that nightly version.
NIGHTLY_VERSION = "2.6.0.dev20241009+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121"
TORCHAO_VERSION = "torchao==0.5.0 --extra-index-url https://download.pytorch.org/whl/cu121"
MEGABLOCKS_VERSION = "megablocks[gg] @ git+https://git@github.com/epwalsh/megablocks.git@epwalsh/deps"
CUDA_TOOLKIT_VERSION = 12.1.0

VERSION = $(shell python src/olmo_core/version.py)
VERSION_SHORT = $(shell python src/olmo_core/version.py short)
IMAGE_BASENAME = olmo-core
BEAKER_WORKSPACE = ai2/OLMo-core
BEAKER_USER = $(shell beaker account whoami --format=json | jq -r '.[0].name')

.PHONY : checks
checks : style-check lint-check type-check

.PHONY : style-check
style-check :
	@echo "======== running isort... ========"
	@isort --check .
	@echo "======== running black... ========"
	@black --check .

.PHONY : lint-check
lint-check :
	@echo "======== running ruff... ========="
	@ruff check .

.PHONY : type-check
type-check :
	@echo "======== running mypy... ========="
	@mypy src/

.PHONY : docs
docs :
	rm -rf docs/build/
	sphinx-autobuild -b html --watch src/olmo_core/ --watch README.md docs/source/ docs/build/

.PHONY : build
build :
	rm -rf *.egg-info/
	python -m build

.PHONY : stable-image
stable-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg BASE=$(BASE_IMAGE) \
		--build-arg CUDA_TOOLKIT_VERSION=$(CUDA_TOOLKIT_VERSION) \
		--build-arg MEGABLOCKS_VERSION=$(MEGABLOCKS_VERSION) \
		--build-arg TORCHAO_VERSION=$(TORCHAO_VERSION) \
		--target stable \
		--progress plain \
		-t $(IMAGE_BASENAME) .
	echo "Built image '$(IMAGE_BASENAME)', size: $$(docker inspect -f '{{ .Size }}' $(IMAGE_BASENAME) | numfmt --to=si)"

.PHONY : nightly-image
nightly-image :
	docker build -f src/Dockerfile \
		--build-arg BUILDKIT_INLINE_CACHE=1 \
		--build-arg BASE=$(BASE_IMAGE) \
		--build-arg CUDA_TOOLKIT_VERSION=$(CUDA_TOOLKIT_VERSION) \
		--build-arg MEGABLOCKS_VERSION=$(MEGABLOCKS_VERSION) \
		--build-arg TORCHAO_VERSION=$(TORCHAO_VERSION) \
		--build-arg NIGHTLY_VERSION=$(NIGHTLY_VERSION) \
		--target nightly \
		--progress plain \
		-t $(IMAGE_BASENAME)-nightly .
	echo "Built image '$(IMAGE_BASENAME)-nightly', size: $$(docker inspect -f '{{ .Size }}' $(IMAGE_BASENAME)-nightly | numfmt --to=si)"

.PHONY : beaker-image-stable
beaker-image-stable : stable-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION_SHORT) $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME) $(IMAGE_BASENAME)-v$(VERSION) $(BEAKER_WORKSPACE)

.PHONY : beaker-image-nightly
beaker-image-nightly : nightly-image
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-nightly $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-v$(VERSION_SHORT)-nightly $(BEAKER_WORKSPACE)
	./src/scripts/beaker/create_beaker_image.sh $(IMAGE_BASENAME)-nightly $(IMAGE_BASENAME)-v$(VERSION)-nightly $(BEAKER_WORKSPACE)

.PHONY : get-beaker-workspace
get-beaker-workspace :
	@echo $(BEAKER_WORKSPACE)

.PHONY : get-full-beaker-image-name
get-full-beaker-image-name :
	@./src/scripts/beaker/get_full_image_name.sh $(IMAGE_NAME) $(BEAKER_WORKSPACE)
