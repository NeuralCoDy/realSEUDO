SUBDIRS := pw common mem sched type table utest wrap app nn store

.PHONY: all mex libraries clean clobber cleandep mktest cleantest test qtest vtest all-subdirs $(SUBDIRS)

include Makefile.inc

OBJECTS := $(foreach dir, $(SUBDIRS), $(wildcard $(dir)/$(BUILDDIR)/*.o))

LIBRARIES := $(BUILDDIR)/$(SHLIB) $(BUILDDIR)/$(ARLIB)

MEXOBJECTS := $(foreach dir, $(SUBDIRS), $(wildcard $(dir)/$(MEXDIR)/*.o))

MEXLIBRARIES := $(MEXDIR)/$(ARLIB)

# need to call make recursively, to reload $(OBJECTS) after they've been built
all: $(SUBDIRS)
	$(MAKE) libraries

libraries: $(LIBRARIES)

$(SUBDIRS):
	$(MAKE) -C $@

mex: 
	for i in $(SUBDIRS) ; do \
		$(MAKE) -C $$i $@ || exit 1; \
	done
	$(MAKE) $(MEXLIBRARIES)

clean clobber::
	for i in $(SUBDIRS) ; do \
		$(MAKE) -C $$i $@ || exit 1; \
	done
	rm -f $(LIBRARIES)

clean::
	rm -rf $(BUILDDIR) $(MEXDIR)
	
test qtest vtest:: mktest

mktest:: all

demo qdemo vdemo:: mkdemo

mkdemo:: all

cleandep mktest cleantest test qtest vtest mkdemo cleandemo demo qdemo vdemo::
	for i in $(SUBDIRS) ; do \
		$(MAKE) -C $$i $@ || exit 1; \
	done

$(LIBRARIES): | $(BUILDDIR)

$(BUILDDIR)/$(SHLIB): $(OBJECTS) 
	rm -f $@
	$(CXX) -shared -o $@ $^

$(BUILDDIR)/$(ARLIB): $(OBJECTS)
	rm -f $@
	$(AR) cr $@ $^
	$(RANLIB) $@

$(MEXLIBRARIES): | $(MEXDIR)

$(MEXDIR)/$(ARLIB): $(MEXOBJECTS)
	rm -f $@
	$(AR) cr $@ $^
	$(RANLIB) $@

