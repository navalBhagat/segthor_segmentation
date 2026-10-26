red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)

SEGTHOR_SOURCE ?= data/segthor_train
SEGTHOR_TEST_SOURCE ?= data/segthor_test
SEGTHOR_SHAPE ?= 256 256
SEGTHOR_RETAINS ?= 5
SEGTHOR_SEED ?= 0
SEGTHOR_FOLD ?= 0
SEGTHOR_PROCESS ?= 1
SEGTHOR_HU_MIN ?= -1000
SEGTHOR_HU_MAX ?= 500
SEGTHOR_MM_PER_PX ?= 0
SEGTHOR_TARGET_SPACING ?= 1.0 1.0 1.0
SEGTHOR_INCLUDE_TEST ?= 0

SEGTHOR_PREPROCESS_ARGS := --shape $(SEGTHOR_SHAPE) \
	--retains $(SEGTHOR_RETAINS) \
	--seed $(SEGTHOR_SEED) \
	--process $(SEGTHOR_PROCESS) \
	--hu_min $(SEGTHOR_HU_MIN) \
	--hu_max $(SEGTHOR_HU_MAX) \
	--mm_per_px $(SEGTHOR_MM_PER_PX) \
	--target_spacing $(SEGTHOR_TARGET_SPACING)

ifeq ($(SEGTHOR_INCLUDE_TEST),1)
SEGTHOR_PREPROCESS_ARGS += --include_test
endif

SEGTHOR_OLD_PREPROCESS_ARGS := --source_dir $(SEGTHOR_SOURCE) \
	--shape $(SEGTHOR_SHAPE) \
	--retains $(SEGTHOR_RETAINS) \
	--seed $(SEGTHOR_SEED) \
	--fold $(SEGTHOR_FOLD) \
	--process $(SEGTHOR_PROCESS)

data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/segthor_train: data/segthor_train.zip
	$(info $(yellow)unzip $<$(reset))
	rm -rf tmp/data/segthor_train
	mkdir -p tmp/data
	unzip -q $< -d tmp/data/segthor_train
	rm -rf $@
	mkdir -p $@
	cp -a tmp/data/segthor_train/. $@
	rm -rf tmp

data/segthor_test: data/test.zip data/test.zip.sha256
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/test.zip.sha256
	mkdir -p $@
	unzip -q $< -d $@
	rm -f $@/.DS_STORE

# New preprocessing pipeline. Supports HU windowing, optional physical crop
# (--mm_per_px in preprocess.py), and optional voxel resampling.
data/SEGTHOR: $(SEGTHOR_SOURCE)
	$(info $(green)python $(CFLAGS) preprocess.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) preprocess.py $(SEGTHOR_PREPROCESS_ARGS) --source_dir $(SEGTHOR_SOURCE) --dest_dir $@_tmp
	mv $@_tmp $@

# Original course preprocessing pipeline, kept separately for comparisons.
data/SEGTHOR_OLD: $(SEGTHOR_SOURCE)
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py $(SEGTHOR_OLD_PREPROCESS_ARGS) --dest_dir $@_tmp
	mv $@_tmp $@

data/SEGTHOR_TEST: $(SEGTHOR_TEST_SOURCE) preprocess.py
	$(info $(green)python $(CFLAGS) preprocess.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) preprocess.py $(SEGTHOR_PREPROCESS_ARGS) --source_dir $(SEGTHOR_TEST_SOURCE) --test_only --dest_dir $@_tmp
	mv $@_tmp $@
