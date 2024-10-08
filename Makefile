# Makefile for GNU ed - The GNU line editor
# Copyright (C) 2006, 2007, 2008, 2009, 2010, 2011, 2012
# Antonio Diaz Diaz.
# This file was generated automatically by configure. Do not edit.
#
# This Makefile is free software: you have unlimited permission
# to copy, distribute and modify it.

pkgname = ed
pkgversion = 1.7
progname = ed
VPATH = .
prefix = /usr/local
exec_prefix = $(prefix)
bindir = $(exec_prefix)/bin
datarootdir = $(prefix)/share
infodir = $(datarootdir)/info
mandir = $(datarootdir)/man
program_prefix = 
CC = gcc
CPPFLAGS = 
CFLAGS = -Wall -W -O3 -march=native -finline-limit=999999999
LDFLAGS = 

#TODO: fts
DISTNAME = $(pkgname)-$(pkgversion)
INSTALL = install
INSTALL_PROGRAM = $(INSTALL) -p -m 755
INSTALL_SCRIPT = $(INSTALL) -p -m 755
INSTALL_DATA = $(INSTALL) -p -m 644
INSTALL_DIR = $(INSTALL) -d -m 755
SHELL = /nix/store/qgwz5zjp1vaskk1v4kvzrwaw9i56k5w0-system-path/bin/sh

objs = buffer.o carg_parser.o global.o io.o \
       main.o main_loop.o regex.o signal.o \
kat/highlight.o kat/hashtable.o 


.PHONY : all install install-bin install-info install-man install-strip \
         uninstall uninstall-bin uninstall-info uninstall-man \
         doc info man check dist clean distclean

all : $(progname) r$(progname)

$(progname) : $(objs)
	$(CC) $(LDFLAGS) -o $@ $(objs) -lm

$(progname)_profiled : $(objs)
	$(CC) $(LDFLAGS) -pg -o $@ $(objs)

r$(progname) : r$(progname).in
	cat $(VPATH)/r$(progname).in > $@
	chmod a+x $@

main.o : main.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -DPROGVERSION=\"$(pkgversion)\" -c -o $@ $< 

%.o : %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $< 

$(objs)       : Makefile ed.h
carg_parser.o : carg_parser.h
main.o        : carg_parser.h


doc : info man

info : $(VPATH)/doc/$(pkgname).info

$(VPATH)/doc/$(pkgname).info : $(VPATH)/doc/$(pkgname).texinfo
	cd $(VPATH)/doc && makeinfo $(pkgname).texinfo

man : $(VPATH)/doc/$(progname).1

$(VPATH)/doc/$(progname).1 : $(progname)
	help2man -n 'line-oriented text editor' \
	  -o $@ ./$(progname)

Makefile : $(VPATH)/configure $(VPATH)/Makefile.in
	./config.status

check : all
	@$(VPATH)/testsuite/check.sh $(VPATH)/testsuite $(pkgversion)

install : install-bin install-info install-man

install-bin : all
	if [ ! -d "$(DESTDIR)$(bindir)" ] ; then $(INSTALL_DIR) "$(DESTDIR)$(bindir)" ; fi
	$(INSTALL_PROGRAM) ./$(progname) "$(DESTDIR)$(bindir)/$(program_prefix)$(progname)"
	$(INSTALL_SCRIPT) ./r$(progname) "$(DESTDIR)$(bindir)/$(program_prefix)r$(progname)"

install-info :
	if [ ! -d "$(DESTDIR)$(infodir)" ] ; then $(INSTALL_DIR) "$(DESTDIR)$(infodir)" ; fi
	$(INSTALL_DATA) $(VPATH)/doc/$(pkgname).info "$(DESTDIR)$(infodir)/$(program_prefix)$(pkgname).info"
	-install-info --info-dir="$(DESTDIR)$(infodir)" "$(DESTDIR)$(infodir)/$(program_prefix)$(pkgname).info"

install-man :
	if [ ! -d "$(DESTDIR)$(mandir)/man1" ] ; then $(INSTALL_DIR) "$(DESTDIR)$(mandir)/man1" ; fi
	$(INSTALL_DATA) $(VPATH)/doc/$(progname).1 "$(DESTDIR)$(mandir)/man1/$(program_prefix)$(progname).1"
	-rm -f "$(DESTDIR)$(mandir)/man1/$(program_prefix)r$(progname).1"
	cd "$(DESTDIR)$(mandir)/man1" && ln -s "$(program_prefix)$(progname).1" "$(program_prefix)r$(progname).1"

install-strip : all
	$(MAKE) INSTALL_PROGRAM='$(INSTALL_PROGRAM) -s' install

uninstall : uninstall-bin uninstall-info uninstall-man

uninstall-bin :
	-rm -f "$(DESTDIR)$(bindir)/$(program_prefix)$(progname)"
	-rm -f "$(DESTDIR)$(bindir)/$(program_prefix)r$(progname)"

uninstall-info :
	-install-info --info-dir="$(DESTDIR)$(infodir)" --remove "$(DESTDIR)$(infodir)/$(program_prefix)$(pkgname).info"
	-rm -f "$(DESTDIR)$(infodir)/$(program_prefix)$(pkgname).info"

uninstall-man :
	-rm -f "$(DESTDIR)$(mandir)/man1/$(program_prefix)$(progname).1"
	-rm -f "$(DESTDIR)$(mandir)/man1/$(program_prefix)r$(progname).1"

dist : doc
	ln -sf $(VPATH) $(DISTNAME)
	tar -cvf $(DISTNAME).tar \
	  $(DISTNAME)/AUTHORS \
	  $(DISTNAME)/COPYING \
	  $(DISTNAME)/ChangeLog \
	  $(DISTNAME)/INSTALL \
	  $(DISTNAME)/Makefile.in \
	  $(DISTNAME)/NEWS \
	  $(DISTNAME)/README \
	  $(DISTNAME)/TODO \
	  $(DISTNAME)/configure \
	  $(DISTNAME)/doc/$(progname).1 \
	  $(DISTNAME)/doc/$(pkgname).info \
	  $(DISTNAME)/doc/$(pkgname).texinfo \
	  $(DISTNAME)/doc/fdl.texinfo \
	  $(DISTNAME)/r$(progname).in \
	  $(DISTNAME)/testsuite/check.sh \
	  $(DISTNAME)/testsuite/*.t \
	  $(DISTNAME)/testsuite/*.d \
	  $(DISTNAME)/testsuite/*.r \
	  $(DISTNAME)/testsuite/*.pr \
	  $(DISTNAME)/testsuite/*.err \
	  $(DISTNAME)/*.h \
	  $(DISTNAME)/*.c
	rm -f $(DISTNAME)
	lzip -v -9 $(DISTNAME).tar

clean :
	-rm -f $(progname) r$(progname) $(progname)_profiled $(objs)

distclean : clean
	-rm -f Makefile config.status *.tar *.tar.lz
