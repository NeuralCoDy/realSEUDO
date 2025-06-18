
.PHONY: mex

CXXFLAGS= -std=c++14 -g -O3 -lm -fPIC -pthread
SOURCES = fista.cpp fista_types.cpp fista_v1.cpp fista_v1_minimizer.cpp fista_gradient.cpp \
	par_fista.cpp draw.cpp seudo.cpp \
	strprintf.cpp ptwrap.cpp
HEADERS = fista.hpp fista_types.hpp fista_v1.hpp fista_v1_minimizer.hpp fista_gradient.hpp \
	par_fista.hpp draw.hpp seudo.hpp \
	strprintf.hpp ptwrap.hpp ptwrap2.hpp
T_SOURCES = test.cpp
T_HEADERS = test.hpp
MEX_BINS = seudo_native.mexa64 # seudo_native_mx.mexa64 seudo_tools.mexa64

OBJECTS = $(SOURCES:.cpp=.o)
T_OBJECTS = $(T_SOURCES:.cpp=.o)

TESTS_V1 = t_fista_v1 t_vsq_fista_v1 t_msq_fista_v1 t_mmsq_fista_v1 t_dsq_fista_v1 # t_txtdraw t_ocr
TESTS = $(TESTS_V1) \
	t_sprite \
	t_fista t_msq_fista t_mmsq_fista t_dsq_fista t_txtdraw t_ocr t_seudo

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
	mex -c -O -outdir $(MEX_OBJDIR) $<

# discover the Matlab library directories, to add them to run path
MEX_LIBPATH=$(shell matlab -e | grep "^LD_LIBRARY_PATH=" | cut -d= -f2 )

ifneq "$(MEX_LIBPATH)" ""
MEX_LIBOPT = -L$(MEX_LIBPATH) -Wl,-rpath=$(MEX_LIBPATH) -Wl,-rpath-link=$(MEX_LIBPATH)
endif

print_libopt:
	echo $(MEX_LIBOPT)

mex: $(MEX_BINS)

clean:
	rm -f core $(TESTS) *.o $(MEX_BINS)
	rm -rf $(MEX_OBJDIR)

test: all
	e=0; for i in $(TESTS); do echo ">>> $$i"; if ! ./$$i; then echo "FAIL"; e=1; fi; done; return $$e

vtest: all
	e=0; for i in $(TESTS); do echo ">>> $$i"; if ! valgrind ./$$i; then echo "FAIL"; e=1; fi; done; return $$e

t_sprite: t_sprite.cpp $(OBJECTS) $(T_OBJECTS)

t_fista: t_fista.cpp $(OBJECTS) $(T_OBJECTS)

t_fista_v1: t_fista_v1.cpp $(OBJECTS) $(T_OBJECTS)

t_vsq_fista_v1: t_vsq_fista_v1.cpp $(OBJECTS) $(T_OBJECTS)

t_msq_fista: t_msq_fista.cpp $(OBJECTS) $(T_OBJECTS)

t_msq_fista_v1: t_msq_fista_v1.cpp $(OBJECTS) $(T_OBJECTS)

t_mmsq_fista: t_mmsq_fista.cpp $(OBJECTS) $(T_OBJECTS)

t_mmsq_fista_v1: t_mmsq_fista_v1.cpp $(OBJECTS) $(T_OBJECTS)

t_dsq_fista: t_dsq_fista.cpp $(OBJECTS) $(T_OBJECTS)

t_dsq_fista_v1: t_dsq_fista_v1.cpp $(OBJECTS) $(T_OBJECTS)

t_txtdraw: t_txtdraw.cpp $(OBJECTS) $(T_OBJECTS)

t_ocr: t_ocr.cpp $(OBJECTS) $(T_OBJECTS)

t_seudo: t_seudo.cpp $(OBJECTS) $(T_OBJECTS)

# There are 2 special things about this rule:
#  * seudo_native.cpp has to be specified as a source file, not a pre-built object file,
#    because otherwise the linker will complain about missing objects referenced from
#    Matlab headers, don't know why
# *  The added linking options make sure that Matlab's shared libraries get used,
#    in particular its version of libstdc++.
%.mexa64: %.cpp $(MEX_OBJECTS)
	# mex LDFLAGS='$$LDFLAGS $(MEX_LIBOPT)' -O -output $@ $< $(MEX_OBJECTS)
	mex -O -output $@ $< $(MEX_OBJECTS)

# another experimental approach
#XXX_seudo_native.mexa64: $(MEX_OBJECTS)
#	mex -n -O -output seudo_native $(MEX_OBJECTS) | perl -e 'while(<STDIN>) {\
#		next if /^Building/;\
#		s/-Wl,-rpath-link,(\S+)/$$1 -Wl,-rpath=$$1/;\
#		print($$_); next; \
#		system($$_) && die "Command failed"; }'
