/*  GNU ed - The GNU line editor.
   Copyright (C) 1993, 1994 Andrew Moore, Talke Studio
   Copyright (C) 2006, 2007, 2008, 2009, 2010, 2011, 2012
   Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "ed.h"
#include <ctype.h>
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "extensions/extension_parser.c"


enum Status
{ QUIT = -1, ERR = -2, EMOD = -3, FATAL = -4 };
//{ DONE = -1, QUIT = -2, ERR = -3, EMOD = -4, FATAL = -5 }; //TODO: why isnt this implemented?

//TODO: remove all warnings generated from gcc since we are standardizing on this compiler
static char def_filename[FILENAME_SIZE] = "";	/* default filename */
static char errmsg[ERROR_MESSAGE_LENGTH] = "";	/* error message buffer */
static char prompt_str[PROMPT_MAX_LENGTH] = "*";	/* command prompt */
static int first_addr = 0, second_addr = 0;
static bool prompt_on = false;	/* if set, show command prompt */
static bool verbose = false;	/* if set, print all error messages */



void
set_def_filename(const char *const s)
{
  strncpy(def_filename, s, sizeof def_filename);
  def_filename[sizeof(def_filename) - 1] = 0;
}

void
set_error_msg(const char *msg)
{
  if (!msg)
    msg = "";
  strncpy(errmsg, msg, sizeof errmsg);
  errmsg[sizeof(errmsg) - 1] = 0;
}

void
set_prompt(const char *const s)
{
  prompt_on = true;
  strncpy(prompt_str, s, sizeof prompt_str);
  prompt_str[sizeof(prompt_str) - 1] = 0;
}

void
set_verbose(void)
{
  verbose = true;
}


//TODO: structure this and clean this up a bit.
//TODO: extract common functionality amongst the markno features
 const line_t *mark[LINE_MARKER_LENGTH];	/* line markers */
static const line_t *second_mark[LINE_MARKER_LENGTH];	/* line markers for tuple marks dependent on mark */
 int markno;		/* line marker count */

static bool
mark_line_node(const line_t *const lp, const line_t *const slp, int c)
{
  c -= 'a';
  if (c < 0 || c >= 26)
    {
      set_error_msg("Invalid mark character");
      return false;
    }
  if (!mark[c])
    ++markno;
  mark[c] = lp;
  if (lp != slp)		//is an address tuple
    {
      second_mark[c] = slp;
    }
  else
    {
      second_mark[c] = 0;
    }
  return true;
}

void
unmark_line_node(const line_t *const lp)
{
  int i;
  for (i = 0; markno && i < 26; ++i)
    if (mark[i] == lp)
      {
	mark[i] = 0;
	second_mark[i] = 0;
	--markno;
      }
}

//TODO: return two for handling k tuples
/* return address of a marked line */
static int
get_marked_node_addr(int c)
{
  c -= 'a';
  if (c < 0 || c >= 26)
    {
      set_error_msg("Invalid mark character");
      return QUIT;
    }
  return get_line_node_addr(mark[c]);
}

/* return address of a secondary marked line */
static int
get_second_marked_node_addr(int c)
{
  c -= 'a';
  if (c < 0 || c >= 26)
    {
      set_error_msg("Invalid mark character");
      return QUIT;
    }
  return get_line_node_addr(second_mark[c]);
}
//TODO: ----------------------------------NUMPAD MARKERS---------------------------------------
static const line_t *mark_num[LINE_MARKER_LENGTH];	/* line markers */
static const line_t *second_mark_num[LINE_MARKER_LENGTH];	/* line markers for tuple marks dependent on mark */
static int markno_num;		/* line marker count */
static bool
mark_num_line_node(const line_t *const lp, const line_t *const second_lp, int c)
{
  if (!mark_num[c])
    ++markno_num;
  mark_num[c] = lp;
  if (lp != second_lp)		//is an address tuple
    {
      second_mark_num[c] = second_lp;
    }
  else
    {
      second_mark_num[c] = 0;
    }
  return true;
}

void
unmark_num_line_node(const line_t *const lp)
{
  int i;
  for (i = 0; markno_num && i < 26; ++i)
    if (mark_num[i] == lp)
      {
	mark_num[i] = 0;
	second_mark_num[i] = 0;
	--markno_num;
      }
}

/* return address of a marked_num line */
static int
get_marked_num_node_addr(int c)
{
  return get_line_node_addr(mark_num[c]);
}

/* return address of a secondary marked_num line */
static int
get_second_marked_num_node_addr(int c)
{
  return get_line_node_addr(second_mark_num[c]);
}
//---------------------------------------------------------------------------------------------

/* Return pointer to copy of shell command in the command buffer */
static const char *
get_shell_command(const char **const ibufpp)
{
  static char *buf = 0;
  static int bufsz = 0;
  static char *shcmd = 0;	/* shell command buffer */
  static int shcmdsz = 0;	/* shell command buffer size */
  static int shcmdlen = 0;	/* shell command length */
  const char *p;		/* substitution char pointer */
  int i = 0, len;

  if (restricted())
    {
      set_error_msg("Shell access restricted");
      return 0;
    }
  if (!get_extended_line(ibufpp, &len, true))
    return 0;
  p = *ibufpp;
  if (!resize_buffer(&buf, &bufsz, len + 1))
    return 0;
  buf[i++] = '!';		/* prefix command w/ bang */
  while (**ibufpp != '\n')
    {
      if (**ibufpp == '!')
	{
	  if (p != *ibufpp)
	    {
	      if (!resize_buffer(&buf, &bufsz, i + 1))
		return 0;
	      buf[i++] = *(*ibufpp)++;
	    }
	  else if (!shcmd || (traditional() && !*(shcmd + 1)))
	    {
	      set_error_msg("No previous command");
	      return 0;
	    }
	  else
	    {
	      if (!resize_buffer(&buf, &bufsz, i + shcmdlen))
		return 0;
	      for (p = shcmd + 1; p < shcmd + shcmdlen;)
		buf[i++] = *p++;
	      p = (*ibufpp)++;
	    }
	}
      else if (**ibufpp == '%')
	{
	  if (!def_filename[0])
	    {
	      set_error_msg("No current filename");
	      return 0;
	    }
	  p = strip_escapes(def_filename);
	  len = strlen(p);
	  if (!resize_buffer(&buf, &bufsz, i + len))
	    return 0;
	  while (len--)
	    buf[i++] = *p++;
	  p = (*ibufpp)++;
	}
      else
	{
	  if (!resize_buffer(&buf, &bufsz, i + 2))
	    return 0;
	  buf[i++] = **ibufpp;
	  if (*(*ibufpp)++ == '\\')
	    buf[i++] = *(*ibufpp)++;
	}
    }
  while (**ibufpp == '\n')
    ++ * ibufpp;		/* skip newline */
  if (!resize_buffer(&shcmd, &shcmdsz, i + 1))
    return 0;
  memcpy(shcmd, buf, i);
  shcmdlen = i;
  shcmd[i] = 0;
  if (*p == '!' || *p == '%')
    printf("%s\n", shcmd + 1);
  return shcmd;
}


static const char *
skip_blanks(const char *p)
{
  while (isspace((unsigned char) *p) && *p != '\n')
    ++p;
  return p;
}


/* Return pointer to copy of filename in the command buffer */
static const char *
get_filename(const char **const ibufpp)
{
  static char *buf = 0;
  static int bufsz = 0;
  const int pmax = path_max(0);
  int n;

  *ibufpp = skip_blanks(*ibufpp);
  if (**ibufpp != '\n')
    {
      int size = 0;
      if (!get_extended_line(ibufpp, &size, true))
	return 0;
      if (**ibufpp == '!')
	{
	  ++*ibufpp;
	  return get_shell_command(ibufpp);
	}
      else if (size > pmax)
	{
	  set_error_msg("Filename too long");
	  return 0;
	}
    }
  else if (!traditional() && !def_filename[0])
    {
      set_error_msg("No current filename");
      return 0;
    }
  if (!resize_buffer(&buf, &bufsz, pmax + 1))
    return 0;
  for (n = 0; **ibufpp != '\n'; ++n, ++*ibufpp)
    buf[n] = **ibufpp;
  buf[n] = 0;
  while (**ibufpp == '\n')
    ++ * ibufpp;		/* skip newline */
  return (may_access_filename(buf) ? buf : 0);
}


static void
invalid_address(void)
{
  set_error_msg("Invalid address");
}


/* return the next line address in the command buffer */
//TODO: add ENUM here once adding the parsers signal feature
static int
next_addr(const char **const ibufpp, int *const addr_cnt)
//TODO: do we still need this?
{
  const char *const s = *ibufpp = skip_blanks(*ibufpp);
  int addr = current_addr();
  bool first = true;		// true == addr, false == offset

  while (true)
    {
      int n;
      const unsigned char ch = **ibufpp;
      if (isdigit(ch))
	{
	  if (!first)
	    {
	      invalid_address();
	      return EMOD;
	    };
	  if (!parse_int(&addr, *ibufpp, ibufpp))
	    return EMOD;
	}
      else
	switch (ch)
	  {
	  case '+':
	  case '\t':
	    //        case ' ':
	  case '-':
	    *ibufpp = skip_blanks(++*ibufpp);
	    if (isdigit((unsigned char) **ibufpp))
	      {
		if (!parse_int(&n, *ibufpp, ibufpp))
		  return EMOD;
		addr += ((ch == '-') ? -n : n);
	      }
	    else if (ch == '+')
	      ++addr;
	    else if (ch == '-')
	      --addr;
	    break;
	  case '.':
	  case '$':
	    if (!first)
	      {
		invalid_address();
		return EMOD;
	      };
	    ++*ibufpp;
	    addr = ((ch == '.') ? current_addr() : last_addr());
	    break;
	    //forward search.
	  case '/':
	    if (!first)
	      {
		// we are using / as an address tuple for numpad mode so kick it back to the parser
		++*addr_cnt;
		return addr;
		break;
	      };
	    addr = next_matching_node_addr(ibufpp, true);
	    if (addr < 0)
	      return EMOD;
	    if (ch == **ibufpp)
	      ++ * ibufpp;
	    break;
	    //backward search.
	  case '?':
	    if (!first)
	      {
		invalid_address();
		return EMOD;
	      };
	    addr = next_matching_node_addr(ibufpp, false);
	    if (addr < 0)
	      return EMOD;
	    if (ch == **ibufpp)
	      ++ * ibufpp;
	    break;
	  case '\'':
	    if (!first)
	      {
		invalid_address();
		return EMOD;
	      };
	    ++*ibufpp;
	    //Check if the markno stores a tuple or just singular line
	    int current_char = (*(*ibufpp)++);
	    int addr_tuple = 0;

	    addr = get_marked_node_addr(current_char);
	    addr_tuple = get_second_marked_node_addr(current_char);
	    if (addr_tuple > 0)
	      {
		first_addr = addr;
		second_addr = addr_tuple;
		++*addr_cnt;
		++*addr_cnt;
		return QUIT;
	      }
	    //check eagerly for a second address markno
	    int is_second = get_marked_node_addr((*(*ibufpp)));
	    if (is_second > 0)
	      {
		(*(*ibufpp)++);
		first_addr = addr;
		second_addr = is_second;
		++*addr_cnt;
		++*addr_cnt;
		//signal we parsed a tuple
		return QUIT;
	      }
	    if (addr < 0)
	      return EMOD;
	    break;
	  case '*':
		//TODO: mark_num_num instead of mark
	    if (!first)
	      {
		invalid_address();
		return EMOD;
	      };
	    ++*ibufpp;
	    //Check if the mark_numno stores a tuple or just singular line
	    int num_current_char = (*(*ibufpp)++);
	    int num_addr_tuple = 0;

	    addr = get_marked_num_node_addr(num_current_char);
	    num_addr_tuple = get_second_marked_num_node_addr(num_current_char);
	    if (num_addr_tuple > 0)
	      {
		first_addr = addr;
		second_addr = num_addr_tuple;
		++*addr_cnt;
		++*addr_cnt;
		return QUIT;
	      }
	    //check eagerly for a second address mark_numno
	    int num_is_second = get_marked_num_node_addr((*(*ibufpp)));
	    if (num_is_second > 0)
	      {
		(*(*ibufpp)++);
		first_addr = addr;
		second_addr = num_is_second;
		++*addr_cnt;
		++*addr_cnt;
		//signal we parsed a tuple
		return QUIT;
	      }
	    if (addr < 0)
	      return EMOD;
	    break;
	  case '%':
	  case ',':
	  case ';':
	    if (first)
	      {
		++*ibufpp;
		++*addr_cnt;
		second_addr = ((ch == ';') ? current_addr() : 1);
		addr = last_addr();
		break;
	      }			/* FALL THROUGH */
	  default:
	    if (*ibufpp == s)
	      return ERR;	/* EOF */
	    if (addr < 0 || addr > last_addr())
	      {
		invalid_address();
		return EMOD;
	      }
	    ++*addr_cnt;
	    return addr;
	  }
      first = false;
    }
}


/* get line addresses from the command buffer until an invalid address
   is seen. Return number of addresses read */
int
extract_addr_range(const char **const ibufpp)
{
  int addr;
  int addr_cnt = 0;

  first_addr = second_addr = current_addr();
  while (true)
    {
      addr = next_addr(ibufpp, &addr_cnt);
      if (addr < QUIT)
	{
	  break;
	}
      else if (addr == QUIT)
	{
	  return (addr_cnt);
	}
      first_addr = second_addr;
      second_addr = addr;
      if (**ibufpp != ';' && **ibufpp != ',' && **ibufpp != ' '
	  && **ibufpp != '/')
	break;
//TODO: unreachable? remove this after lints
      if (**ibufpp == ';')
	set_current_addr(addr);
      ++*ibufpp;
    }
  if (addr_cnt == 1 || second_addr != addr)
    first_addr = second_addr;
  return ((addr != EMOD) ? addr_cnt : ERR);
}


/* get a valid address from the command buffer */
static bool
get_third_addr(const char **const ibufpp, int *const addr)
{
  const int old1 = first_addr;
  const int old2 = second_addr;
  int addr_cnt = extract_addr_range(ibufpp);

  if (addr_cnt < 0)
    return false;
  if (traditional() && addr_cnt == 0)
    {
      set_error_msg("Destination expected");
      return false;
    }
  if (second_addr < 0 || second_addr > last_addr())
    {
      invalid_address();
      return false;
    }
  *addr = second_addr;
  first_addr = old1;
  second_addr = old2;
  return true;
}


/* return true if address range is valid */
static bool
check_addr_range(const int n, const int m, const int addr_cnt)
{
  if (addr_cnt == 0)
    {
      first_addr = n;
      second_addr = m;
    }
  if (first_addr < 1 || first_addr > second_addr || second_addr > last_addr())
    {
      invalid_address();
      return false;
    }
  return true;
}


/* return true if current address is valid */
static bool
check_current_addr(const int addr_cnt)
{
  return check_addr_range(current_addr(), current_addr(), addr_cnt);
}


/* verify the command suffix in the command buffer */
static bool
get_command_suffix(const char **const ibufpp, int *const gflagsp)
{
  while (true)
    {
      const char ch = **ibufpp;
      //TODO: switch this up
      if (ch == 'l')
	*gflagsp |= GLS;
      else if (ch == 'n')
	*gflagsp |= GNP;
      else if (ch == 'p')
	*gflagsp |= GPR;
      else
	break;
      ++*ibufpp;
    }
  if (*(*ibufpp)++ != '\n')
    {
      set_error_msg("Invalid command suffix");
      return false;
    }
  return true;
}


static bool
unexpected_address(const int addr_cnt)
{
  if (addr_cnt > 0)
    {
      set_error_msg("Unexpected address");
      return true;
    }
  return false;
}

static bool
unexpected_command_suffix(const unsigned char ch)
{
  if (!isspace(ch))
    {
      set_error_msg("Unexpected command suffix");
      return true;
    }
  return false;
}


static bool
command_s(const char **const ibufpp, int *const gflagsp,
	  const int addr_cnt, const bool isglobal)
{
  static int gflags = 0;
  static int snum = 0;
  enum Sflags
  {
    SGG = 0x01,			/* complement previous global substitute suffix */
    SGP = 0x02,			/* complement previous print suffix */
    SGR = 0x04,			/* use last regex instead of last pat */
    SGF = 0x08			/* repeat last substitution */
  } sflags = 0;

  do
    {
      if (isdigit((unsigned char) **ibufpp))
	{
	  if (!parse_int(&snum, *ibufpp, ibufpp))
	    return false;
	  sflags |= SGF;
	  gflags &= ~GSG;	/* override GSG */
	}
      else
	switch (**ibufpp)
	  {
	  case '\n':
	    sflags |= SGF;
	    break;
	  case 'g':
	    sflags |= SGG;
	    ++*ibufpp;
	    break;
	  case 'p':
	    sflags |= SGP;
	    ++*ibufpp;
	    break;
	  case 'r':
	    sflags |= SGR;
	    ++*ibufpp;
	    break;
	  default:
	    if (sflags)
	      {
		set_error_msg("Invalid command suffix");
		return false;
	      }
	  }
    }
  while (sflags && **ibufpp != '\n');
  if (sflags && !prev_pattern())
    {
      set_error_msg("No previous substitution");
      return false;
    }
  if (sflags & SGG)
    snum = 0;			/* override numeric arg */
  if (**ibufpp != '\n' && (*ibufpp)[1] == '\n')
    {
      set_error_msg("Invalid pattern delimiter");
      return false;
    }
  if ((!sflags || (sflags & SGR)) && !new_compiled_pattern(ibufpp))
    return false;
  if (!sflags && !extract_subst_tail(ibufpp, &gflags, &snum, isglobal))
    return false;
  if (isglobal)
    gflags |= GLB;
  else
    gflags &= ~GLB;
  if (sflags & SGG)
    gflags ^= GSG;
  if (sflags & SGP)
    {
      gflags ^= GPR;
      gflags &= ~(GLS | GNP);
    }
  switch (**ibufpp)
    {
    case 'l':
      gflags |= GLS;
      ++*ibufpp;
      break;
    case 'n':
      gflags |= GNP;
      ++*ibufpp;
      break;
    case 'p':
      gflags |= GPR;
      ++*ibufpp;
      break;
    }
  if (!check_current_addr(addr_cnt) || !get_command_suffix(ibufpp, gflagsp))
    return false;
  if (!isglobal)
    clear_undo_stack();
  if (!search_and_replace(first_addr, second_addr, gflags, snum, isglobal))
    return false;
  if ((gflags & (GPR | GLS | GNP)) &&
      !display_lines(current_addr(), current_addr(), gflags))
    return false;
  return true;
}


//NOTE: skeleton
static bool exec_global(const char **const ibufpp,
			const int gflags, const bool interactive);

/* execute the next command in command buffer; return error status */
static int
exec_command(const char **const ibufpp,
	     const int prev_status, const bool isglobal)
{
  const char *fnp;
  int gflags = 0;
  int addr, c, n;

  const int addr_cnt = extract_addr_range(ibufpp);
  if (addr_cnt < 0)
    return ERR;
  *ibufpp = skip_blanks(*ibufpp);
  c = *(*ibufpp)++;
  switch (c)
    {
    case '`':
      //TODO: extensions dont allow command list, we need to ensure were doing everything idiomatically we need all the features such as writting the output in extensions
      //TODO: handle the proper idiomatic address etc errors
      fnp = def_filename;
      parse_extension(ibufpp, fnp, &first_addr, &second_addr);
      break;
    case 'a':
      if (!get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!append_lines(ibufpp, second_addr, isglobal))
	return ERR;
      break;
    case 'c':
      if (first_addr == 0)
	first_addr = 1;
      if (second_addr == 0)
	second_addr = 1;
      if (!check_current_addr(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!delete_lines(first_addr, second_addr, isglobal) ||
	  !append_lines(ibufpp, current_addr(), isglobal))
	return ERR;
      break;
    case 'd':
      if (!check_current_addr(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!delete_lines(first_addr, second_addr, isglobal))
	return ERR;
      inc_current_addr();
      break;
    case 'e':
      if (modified() && !scripted() && prev_status != EMOD)
	return EMOD;		/* fall through */
    case 'E':
      if (unexpected_address(addr_cnt) || unexpected_command_suffix(**ibufpp))
	return ERR;
      fnp = get_filename(ibufpp);
      if (!fnp || !delete_lines(1, last_addr(), isglobal) || !close_sbuf())
	return ERR;
      if (!open_sbuf())
	return FATAL;
      if (fnp[0] && fnp[0] != '!')
	set_def_filename(fnp);
      if (traditional() && !fnp[0] && !def_filename[0])
	{
	  set_error_msg("No current filename");
	  return ERR;
	}
      if (read_file(fnp[0] ? fnp : def_filename, 0) < 0)
	return ERR;
      reset_undo_state();
      set_modified(false);
      break;
    case 'f':
      if (unexpected_address(addr_cnt) || unexpected_command_suffix(**ibufpp))
	return ERR;
      fnp = get_filename(ibufpp);
      if (!fnp)
	return ERR;
      if (fnp[0] == '!')
	{
	  set_error_msg("Invalid redirection");
	  return ERR;
	}
      if (fnp[0])
	set_def_filename(fnp);
      printf("%s\n", strip_escapes(def_filename));
      break;
    case 'g':
    case 'v':
    case 'G':
    case 'V':
      if (isglobal)
	{
	  set_error_msg("Cannot nest global commands");
	  return ERR;
	}
      n = (c == 'g' || c == 'G');	/* mark matching lines */
      if (!check_addr_range(1, last_addr(), addr_cnt) ||
	  !build_active_list(ibufpp, first_addr, second_addr, n))
	return ERR;
      n = (c == 'G' || c == 'V');	/* interactive */
      if ((n && !get_command_suffix(ibufpp, &gflags)) ||
	  !exec_global(ibufpp, gflags, n))
	return ERR;
      break;
    case 'h':
    case 'H':
      if (unexpected_address(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (c == 'H')
	verbose = !verbose;
      if ((c == 'h' || verbose) && errmsg[0])
	fprintf(stderr, "%s\n", errmsg);
      break;
    case 'i':
      if (second_addr == 0)
	second_addr = 1;
      if (!get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!append_lines(ibufpp, second_addr - 1, isglobal))
	return ERR;
      break;
    case 'j':
      if (!check_addr_range(current_addr(), current_addr() + 1, addr_cnt)
	  || !get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (first_addr != second_addr &&
	  !join_lines(first_addr, second_addr, isglobal))
	return ERR;
      break;
    case ']':
      n = *(*ibufpp)++;
      if (second_addr == 0)
	{
	  invalid_address();
	  return ERR;
	}
      if (!get_command_suffix(ibufpp, &gflags) ||
	  !mark_num_line_node(search_line_node(first_addr),
			  search_line_node(second_addr), n))
	return ERR;
      break;
    case 'k':
      n = *(*ibufpp)++;
      if (second_addr == 0)
	{
	  invalid_address();
	  return ERR;
	}
      if (!get_command_suffix(ibufpp, &gflags) ||
	  !mark_line_node(search_line_node(first_addr),
			  search_line_node(second_addr), n))
	//        !mark_line_node(search_line_node(second_addr), n))//NOTE: ORIGINAL
	return ERR;
      break;
    case 'l':
    case 'n':
    case 'p':
      if (c == 'l')
	n = GLS;
      else if (c == 'n')
	n = GNP;
      else
	n = GPR;
      if (!check_current_addr(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags) ||
	  !display_lines(first_addr, second_addr, gflags | n))
	return ERR;
      gflags = 0;
      break;
    case 'm':
      if (!check_current_addr(addr_cnt) || !get_third_addr(ibufpp, &addr))
	return ERR;
      if (addr >= first_addr && addr < second_addr)
	{
	  set_error_msg("Invalid destination");
	  return ERR;
	}
      if (!get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!move_lines(first_addr, second_addr, addr, isglobal))
	return ERR;
      break;
    case 'P':
    case 'q':
    case 'Q':
      if (unexpected_address(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (c == 'P')
	prompt_on = !prompt_on;
      else if (modified() && !scripted() && c == 'q' && prev_status != EMOD)
	return EMOD;
      else
	return QUIT;
      break;
    case 'r':
      if (unexpected_command_suffix(**ibufpp))
	return ERR;
      if (addr_cnt == 0)
	second_addr = last_addr();
      fnp = get_filename(ibufpp);
      if (!fnp)
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!def_filename[0] && fnp[0] != '!')
	set_def_filename(fnp);
      if (traditional() && !fnp[0] && !def_filename[0])
	{
	  set_error_msg("No current filename");
	  return ERR;
	}
      addr = read_file(fnp[0] ? fnp : def_filename, second_addr);
      if (addr < 0)
	return ERR;
      if (addr)
	set_modified(true);
      break;
    case 's':
      if (!command_s(ibufpp, &gflags, addr_cnt, isglobal))
	return ERR;
      break;
    case 't':
      if (!check_current_addr(addr_cnt) ||
	  !get_third_addr(ibufpp, &addr) ||
	  !get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!copy_lines(first_addr, second_addr, addr))
	return ERR;
      break;
    case 'u':
      if (unexpected_address(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags) || !undo(isglobal))
	return ERR;
      break;
    case 'w':
    case 'W':
      n = **ibufpp;
      if (n == 'q' || n == 'Q')
	++ * ibufpp;
      if (unexpected_command_suffix(**ibufpp))
	return ERR;
      fnp = get_filename(ibufpp);
      if (!fnp)
	return ERR;
      if (addr_cnt == 0 && last_addr() == 0)
	first_addr = second_addr = 0;
      else if (!check_addr_range(1, last_addr(), addr_cnt))
	return ERR;
      if (!def_filename[0] && fnp[0] != '!')
	set_def_filename(fnp);
      if (traditional() && !fnp[0] && !def_filename[0])
	{
	  set_error_msg("No current filename");
	  return ERR;
	}
      addr = write_file(fnp[0] ? fnp : def_filename,
			(c == 'W') ? "a" : "w", first_addr, second_addr);
      if (addr < 0)
	return ERR;
      if (addr == last_addr())
	set_modified(false);
      else if (modified() && !scripted() && n == 'q' && prev_status != EMOD)
	return EMOD;
      if (n == 'q' || n == 'Q')
	return QUIT;
      break;
    case 'x':
      if (second_addr < 0 || last_addr() < second_addr)
	{
	  invalid_address();
	  return ERR;
	}
      if (!get_command_suffix(ibufpp, &gflags))
	return ERR;
      if (!isglobal)
	clear_undo_stack();
      if (!put_lines(second_addr))
	return ERR;
      break;
    case 'y':
      if (!check_current_addr(addr_cnt) ||
	  !get_command_suffix(ibufpp, &gflags) ||
	  !yank_lines(first_addr, second_addr))
	return ERR;
      break;
    case 'z':
      first_addr = 1;
      if (!check_addr_range(first_addr, current_addr() +
			    (traditional() || !isglobal), addr_cnt))
	return ERR;
      if (**ibufpp > '0' && **ibufpp <= '9')
	{
	  if (parse_int(&n, *ibufpp, ibufpp))
	    set_window_lines(n);
	  else
	    return ERR;
	}
      if (!get_command_suffix(ibufpp, &gflags) ||
	  !display_lines(second_addr,
			 min(last_addr(), second_addr + window_lines()),
			 gflags))
	return ERR;
      gflags = 0;
      break;
    case '=':
      if (!get_command_suffix(ibufpp, &gflags))
	return ERR;
      printf("%d\n", addr_cnt ? second_addr : last_addr());
      break;
    case '!':
      if (unexpected_address(addr_cnt))
	return ERR;
      fnp = get_shell_command(ibufpp);
      if (!fnp)
	return ERR;
      if (system(fnp + 1) < 0)
	{
	  set_error_msg("Can't create shell process");
	  return ERR;
	}
      if (!scripted())
	printf("!\n");
      break;
    case '\n':
      //TODO: test and debug, ensure this if propogates idiomatically like the else
      if (first_addr != second_addr)
	{
	  if (!display_lines(first_addr, second_addr, 0))
	    return ERR;
	}
      else
	{
	  first_addr = 1;
	  if (!check_addr_range
	      (first_addr, current_addr() + (traditional() || !isglobal),
	       addr_cnt) || !display_lines(second_addr, second_addr, 0))
	    return ERR;
	  return 0;
	}
      break;
    case '#':
      while (*(*ibufpp)++ != '\n');
      break;
    default:
      set_error_msg("Unknown command");
      return ERR;
    }
  if (gflags && !display_lines(current_addr(), current_addr(), gflags))
    return ERR;

  return 0;
}


/* apply command list in the command buffer to the active lines in a
   range; return false if error */
static bool
exec_global(const char **const ibufpp,
	    const int gflags, const bool interactive)
{
  static char *buf = 0;
  static int bufsz = 0;
  const char *cmd = 0;

  if (!interactive)
    {
      if (traditional() && !strcmp(*ibufpp, "\n"))
	cmd = "p\n";		/* null cmd_list == 'p' */
      else
	{
	  if (!get_extended_line(ibufpp, 0, false))
	    return false;
	  cmd = *ibufpp;
	}
    }
  clear_undo_stack();
  while (true)
    {
      const line_t *const lp = next_active_node();
      if (!lp)
	break;
      set_current_addr(get_line_node_addr(lp));
      if (current_addr() < 0)
	return false;
      if (interactive)
	{
	  /* print current_addr; get a command in global syntax */
	  int len;
	  if (!display_lines(current_addr(), current_addr(), gflags))
	    return false;
	  do
	    {
	      *ibufpp = get_tty_line(&len);
	    }
	  while (*ibufpp && len > 0 && (*ibufpp)[len - 1] != '\n');
	  if (!*ibufpp)
	    return false;
	  if (len == 0)
	    {
	      set_error_msg("Unexpected end-of-file");
	      return false;
	    }
	  if (len == 1 && !strcmp(*ibufpp, "\n"))
	    continue;
	  if (len == 2 && !strcmp(*ibufpp, "&\n"))
	    {
	      if (!cmd)
		{
		  set_error_msg("No previous command");
		  return false;
		}
	    }
	  else
	    {
	      if (!get_extended_line(ibufpp, &len, false) ||
		  !resize_buffer(&buf, &bufsz, len + 1))
		return false;
	      memcpy(buf, *ibufpp, len + 1);
	      cmd = buf;
	    }
	}
      *ibufpp = cmd;
      while (**ibufpp)
	if (exec_command(ibufpp, 0, true) < 0)
	  return false;
    }
  return true;
}


//TODO: free routine on exit for ibufp_prev and maybe ibufp, not really necessary since Im pretty sure they leak and let linux process free but I'd like to free correctly with a valgrind check
//TODO: add simple features like a macro system for storing commands that can be executed with string substitution
//TODO: allow ed to manage reading in multiple files as one large **line_t
int
main_loop(const bool loose)
{
  //TODO: store macros here for templating ibufp commands, this will solve a lot of repetition.
  char *theme_file = NULL;
  highlight_init(theme_file);
  sharpie_alloc();
  //TODO: also initialize a line for the highlighter since we may not be getting correct highlight state given highlighter errors
  extern jmp_buf jmp_state;
  const char *ibufp = NULL;	/* pointer to command buffer */
  const char *ibufp_prev = NULL;	/* pointer to previous command buffer */
  volatile int err_status = 0;	/* program exit status */
  volatile int linenum = 0;	/* script line number */
  int len, status = 0;

  disable_interrupts();
  set_signals();
  status = setjmp(jmp_state);
  if (!status)
    enable_interrupts();
  else
    {
      status = QUIT;
      fputs("\n?\n", stderr);
      set_error_msg("Interrupt");
    }

  while (true)
    {
      fflush(stdout);
      if (status < 0 && verbose)
	{
	  fprintf(stderr, "%s\n", errmsg);
	  fflush(stderr);
	}
      if (prompt_on)
	{
	  printf("%s", prompt_str);
	  fflush(stdout);
	}
      ibufp = get_tty_line(&len);
      if (!ibufp)
	return err_status;
      if (!len)
	{
	  if (!modified() || scripted())
	    return err_status;
	  fputs("?\n", stderr);
	  set_error_msg("Warning: buffer modified");
	  if (is_regular_file(0))
	    {
	      if (verbose)
		fprintf(stderr, "script, line %d: %s\n", linenum, errmsg);
	      return 2;
	    }
	  set_modified(false);
	  status = EMOD;
	  continue;
	}
      else if (ibufp[len - 1] != '\n')	/* discard line */
	{
	  set_error_msg("Unexpected end-of-file");
	  status = ERR;
	  continue;
	}
      ++linenum;
      //if only a newline, repeat last command.
      if (len == 1)
	{
	  ibufp = ibufp_prev;
	  status = exec_command(&ibufp, status, false);
	}
      else
	{
	  ibufp_prev =
	    realloc(ibufp_prev, (1 + strlen(ibufp)) * sizeof(char));
	  strcpy(ibufp_prev, ibufp);
	  status = exec_command(&ibufp, status, false);
	}
      if (status == 0)
	continue;
      if (status == QUIT)
	return err_status;
      if (status == EMOD)
	{
	  fputs("?\n", stderr);	/* give warning */
	  set_error_msg("Warning: buffer modified");
	  if (is_regular_file(0))
	    {
	      if (verbose)
		fprintf(stderr, "script, line %d: %s\n", linenum, errmsg);
	      return 1;
	    }
	}
      else if (status == FATAL)
	{
	  if (verbose)
	    {
	      if (is_regular_file(0))
		fprintf(stderr, "script, line %d: %s\n", linenum, errmsg);
	      else
		fprintf(stderr, "%s\n", errmsg);
	    }
	  return 1;
	}
      else
	{
	  fputs("?\n", stderr);	/* give warning */
	  if (is_regular_file(0))
	    {
	      if (verbose)
		fprintf(stderr, "script, line %d: %s\n", linenum, errmsg);
	      return 1;
	    }
	}
      if (!loose)
	err_status = 1;
    }
}
