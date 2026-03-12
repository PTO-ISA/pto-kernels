# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# All rights reserved.
# See LICENSE in the root of the software repository:
# https://github.com/huawei-csl/pto-kernels/
# for the full License text.
# --------------------------------------------------------------------------------
.PHONY: clean setup_once build_wheel install test bootstrap check-env sync-skills bench-dry-run

clean:
	rm -rf build/ dist/ extra-info/ *.egg-info/ kernel_meta/

setup_once:
	pip3 install -r requirements.txt
	pip3 install torch-npu==2.8.0.post2 --extra-index-url https://download.pytorch.org/whl/cpu

build_cmake: clean
	bash scripts/build.sh

build_wheel:
	export CMAKE_GENERATOR="Unix Makefiles" && pip wheel -v  . --extra-index-url https://download.pytorch.org/whl/cpu

install: build_wheel
	bash scripts/install-wheel.sh

test:
	pytest -v tests/

bootstrap:
	bash scripts/bootstrap_workspace.sh

check-env:
	bash -lc 'source scripts/source_env.sh && python3 scripts/check_env.py --strict'

sync-skills:
	bash scripts/install_codex_skills.sh

bench-dry-run:
	PYTHONPATH=python python3 -m pto_kernels.bench.runner bench/specs/posembedding/apply_rotary_pos_emb.yaml --dry-run
