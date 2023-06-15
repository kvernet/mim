# Compiler flags
CC       = gcc
CFLAGS   = -O0 -g -Wall -std=c99
LIBNAME  = mim
ULIBNAME = MIM

# OS dependent flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SOEXT = dylib
	LIB   = lib$(LIBNAME).$(SOEXT)
	LD    = $(CC) -dynamiclib -Wl,-install_name,@rpath/$(LIB)
	RPATH = -Wl,-rpath,@loader_path/../lib
else
	SOEXT = so
	LIB   = lib$(LIBNAME).$(SOEXT)
	LD    = $(CC) -shared
	RPATH = '-Wl,-rpath,$$ORIGIN/../lib'
endif

# Version name
VERSION_MAJOR  = $(shell grep $(ULIBNAME)_VERSION_MAJOR src/$(LIBNAME).h | cut -d' ' -f3)
VERSION_MINOR  = $(shell grep $(ULIBNAME)_VERSION_MINOR src/$(LIBNAME).h | cut -d' ' -f3)
VERSION_PATCH  = $(shell grep $(ULIBNAME)_VERSION_PATCH src/$(LIBNAME).h | cut -d' ' -f3)

LIB_SHORTNAME  = $(LIB).$(VERSION_MAJOR)
LIB_FULLNAME   = $(LIB_SHORTNAME).$(VERSION_MINOR).$(VERSION_PATCH)

# C library compilation flag
LIB_CFLAGS  = -fPIC $(CFLAGS)

.PHONY: lib

lib: lib/$(LIB_FULLNAME) \
	lib/$(LIB_SHORTNAME) \
	lib/$(LIB)

lib/$(LIB_FULLNAME): src/$(LIBNAME).c src/$(LIBNAME).h | libdir
	$(LD) -o $@ $(LIB_CFLAGS) $^

lib/$(LIB_SHORTNAME): lib/$(LIB_FULLNAME)
	@ln -fs $(LIB_FULLNAME) $@

lib/$(LIB): lib/$(LIB_SHORTNAME)
	@ln -fs $(LIB_SHORTNAME) $@

.PHONY: libdir

libdir:
	@mkdir -p lib

# C examples compilation
examples: bin/img

EXMAPLES_CFLAGS = $(CFLAGS) -Isrc
EXAMPLES_LDFLAGS = -L$(PWD)/lib -Wl,-rpath,$(PWD)/lib -l$(LIBNAME)

bin/%: examples/%.c src/$(LIBNAME).h | lib/$(LIB) bindir
	$(CC) $(EXMAPLES_CFLAGS) -o $@ $< $(EXAMPLES_LDFLAGS)

.PHONY: bindir
bindir:
	@mkdir -p bin


# Cleaning
.PHONY: clean

clean:
	rm -rf lib
	rm -rf bin
