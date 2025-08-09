# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -W
SOURCEDIR     = source
BUILDDIR      = build

USE_UV ?= 0
HAS_UV := $(shell command -v uv >/dev/null 2>&1 && echo 1 || echo 0)

ifeq ($(USE_UV),1)
  ifeq ($(HAS_UV),1)
    SPHINXBUILD := uv run --no-project sphinx-build
  else
    $(error "USE_UV=1 but uv not found in PATH")
  endif
else
  SPHINXBUILD := sphinx-build
endif

# Put it first so that "make" without argument is like "make help".
help:
	$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
