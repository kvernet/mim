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
	MLIB  = -lm
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

# Python package.
PYTHON=  python3
PACKAGE= wrapper.abi3.$(SOEXT)
OBJS=    src/wrapper.o

.PHONY: package
package: mim/$(PACKAGE) \
         mim/lib/$(LIB) \
         mim/include/mim.h

mim/$(PACKAGE): setup.py src/build-wrapper.py $(OBJS) lib/$(LIB)
	$(PYTHON) setup.py build --build-lib .
	@rm -rf build mim.egg-info

src/%.o: src/%.c src/%.h
	$(CC) $(LIB_CFLAGS) -c -o $@ $<

mim/lib/$(LIB): lib/$(LIB)
	@mkdir -p mim/lib
	@ln -fs ../../$< $@

mim/include/%.h: src/%.h
	@mkdir -p mim/include
	@ln -fs ../../$< $@

# C examples compilation
examples: bin/img \
			bin/model \
			bin/stat

EXMAPLES_CFLAGS = $(CFLAGS) -Isrc
EXAMPLES_LDFLAGS = -L$(PWD)/lib -Wl,-rpath,$(PWD)/lib -l$(LIBNAME) $(MLIB)

bin/%: examples/%.c src/$(LIBNAME).h | lib/$(LIB) bindir
	$(CC) $(EXMAPLES_CFLAGS) -o $@ $< $(EXAMPLES_LDFLAGS)

.PHONY: bindir
bindir:
	@mkdir -p bin


# Cleaning
.PHONY: clean

clean:
	rm -rf bin
	rm -rf build
	rm -rf lib
	rm -f src/*.o
	rm -rf mim/$(PACKAGE) mim/__pycache__ mim/version.py
	rm -rf mim/include mim/lib
