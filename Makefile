.PHONY: format format-c

CLANG_FORMAT ?= clang-format
FORMAT_OPTS += -style=file -i

format-c :
	$(CLANG_FORMAT) $(FORMAT_OPTS) \
          swe/*.c

format : format-c


