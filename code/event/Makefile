
.PHONY: all mex clean test vtest triceps

INCFLAGS := -Itriceps
LIBTRICEPS := triceps/mexbuild/libtriceps.a
LIBFLAGS := $(LIBTRICEPS)
CXXFLAGS := -std=c++14 -g -O3 -lm -fPIC -pthread $(INCFLAGS)
SOURCES := event_table.cpp
HEADERS := event_table.hpp
T_SOURCES := test.cpp
T_HEADERS := test.hpp
MEX_BINS := rois_event_native.mexa64

OBJECTS = $(SOURCES:.cpp=.o)
T_OBJECTS = $(T_SOURCES:.cpp=.o)

TESTS = t_table

MEX_OBJDIR = mexobj
MEX_OBJECTS = $(addprefix $(MEX_OBJDIR)/, $(SOURCES:.cpp=.o))

all: $(TESTS)

# The object files are built separately for mex because they might be of
# a different architecture, for example on Mac with an ARM CPU Matlab
# is still compiled for x86, so the mex files get compiled for x86 too.
$(MEX_OBJDIR):
	mkdir $(MEX_OBJDIR)

$(MEX_OBJECTS): | $(MEX_OBJDIR)

$(MEX_OBJDIR)/%.o: %.cpp # Makefile
	mex $(INCFLAGS) -c -g -O -outdir $(MEX_OBJDIR) $<

# discover the Matlab library directories, to add them to run path
MEX_LIBPATH=$(shell matlab -e | grep "^LD_LIBRARY_PATH=" | cut -d= -f2 )

ifneq "$(MEX_LIBPATH)" ""
MEX_LIBOPT = -L$(MEX_LIBPATH) -Wl,-rpath=$(MEX_LIBPATH) -Wl,-rpath-link=$(MEX_LIBPATH)
endif

print_libopt:
	echo $(MEX_LIBOPT)

triceps:
	$(MAKE) -C triceps mex

mex: triceps $(MEX_BINS)

clean:
	rm -f core $(TESTS) *.o $(MEX_BINS)
	rm -rf $(MEX_OBJDIR)
	$(MAKE) -C triceps clean

test: all
	e=0; for i in $(TESTS); do echo ">>> $$i"; if ! ./$$i; then echo "FAIL"; e=1; fi; done; return $$e

vtest: all
	e=0; for i in $(TESTS); do echo ">>> $$i"; if ! valgrind ./$$i; then echo "FAIL"; e=1; fi; done; return $$e

t_table: t_table.cpp $(OBJECTS) $(T_OBJECTS) $(LIBTRICEPS)

# There are 2 special things about this rule:
#  * rois_event_native.cpp has to be specified as a source file, not a pre-built object file,
#    because otherwise the linker will complain about missing objects referenced from
#    Matlab headers, don't know why
# *  The added linking options make sure that Matlab's shared libraries get used,
#    in particular its version of libstdc++.
%.mexa64: %.cpp $(MEX_OBJECTS)
	# mex LDFLAGS='$$LDFLAGS $(MEX_LIBOPT)' -O -output $@ $< $(MEX_OBJECTS)
	mex $(INCFLAGS) -O -g -output $@ $< $(MEX_OBJECTS) $(LIBFLAGS)

