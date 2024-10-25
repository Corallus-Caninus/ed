#include <string.h>
//TODO: consider gperf for this
//TODO: libgit automation to squash stuff so writes arent so dangerous in ed
//TODO: calling this extensions is a bit confusing given the info file. consider "mods" or "extensions"
//TODO: extract to config? this should be automated but if manually entered it should be extracted.

/* EXTENSION PROTOTPYES */
void file(const char *);
//TODO: do we need to pass pointer to pointer?
void funcs(const char **const, const char *, int **, int **);
void entry(const char **const, int **first_addr, int **second_addr);
void marknos(const char **const, int **first_addr, int **second_addr);

//TODO: make this faster, for now this at least short circuits hopefully faster than strcmp alternative
bool
faststrcmp(const char *const key, const char *const lookup, int len)
{
  int cmp = 0;
  for (int i = 0; i < len; i++)
    {
      cmp = key[i] ^ lookup[i];
      switch (cmp)
	{
	case 0:
	  break;		//or continue;? which is faster
	default:
	  return 0;
	}
    }
  return 1;
}

/* CALL TABLE */
//TODO: structure this with length to make faststrcmp faster (not requiring strlen calls)
static const char *extensions_s[] = {
  "file\n",
  "funcs\n",
  "entry\n",
  "mos\n",
};

/* JUMP TABLE EXTENSION ENUMERATION */
// These are ordered exactly as in extensions_s
#define  NUM_EXTENSIONS 4
static const enum Extension
{
  FILENAME = 1,			// start at 1 so 0 is default "not found" in switch statement
  FUNCS,
  ENTRY,
  MARKNOS,
};

/* PARSER */
//TODO: aint no point in passing in ibufpp, we switched already on the rest of the string, unless we strcmp for size which doesnt seem right.
//Extensions should have argument format: inline int EXTENSION_NAME(const char ** const ibufpp, int *first_addr, int *second_addr, char* args)
int
parse_extension(const char **const ibufpp, const char *filename,
		int *first_addr, int *second_addr)
{
  //  printf ("%s %i, %i \n", *ibufpp, *first_addr, *second_addr);        //TODO: remove after testing

  //find the extension enum from the call table given the ibufpp 
  int extension_index = 0;
  int len;
  int buff_len = strlen(*ibufpp);
  enum Extension extension_select;
  for (int i = 1; i <= NUM_EXTENSIONS; i++)
    //TODO: resort extensions_s by recency?
    {
      len = strlen(extensions_s[i - 1]);	//TODO: we can know this statically with another static global array
      if (buff_len < len)
	len = buff_len;
      if (faststrcmp(extensions_s[i - 1], *ibufpp, len))
	{
	  extension_index = i;
	  break;
	}
    }
  extension_select = (enum Extension) extension_index;

  //NOTE: currently we pass by reference the commands in case we want to fallthrough and reuse the pointers.
  //call the extension given the selection, this can be used for error handling and data preperation as in main_loop.c enum
  switch (extension_select)
    {
    case FILENAME:
      file(filename);
      break;
    case FUNCS:
      funcs(ibufpp, filename, &first_addr, &second_addr);
      entry(ibufpp, &first_addr, &second_addr);
      break;
    case ENTRY:
      entry(ibufpp, &first_addr, &second_addr);
      break;
    case MARKNOS:
      marknos(ibufpp, &first_addr, &second_addr);
      break;
    default:
      printf("extension not found..\n");
    }
  return 0;			//TODO: error codes, also within cases
}

/* BEGIN EXTENSION DECLARATIONS */

//uses ctags c library to list all funcs from first_addr to second_addr in ibufpp
//TODO: remember ibuf is command parsed string, the line_t buffer is global
inline void
funcs(const char **const ibufpp, const char *filename, int **first_addr,
      int **second_addr)
{
  printf("inside funcs\n");
  printf(filename);
}

inline void
file(const char *filename)
{
  printf("%s \n", filename);
}
inline void
prompt(const char*prompt)
{
  printf("%s \n", prompt);
}

inline void
entry(const char **const ibufpp, int **first_addr, int **second_addr)
{
  char *mystring = "test\n";
  int len = strlen(mystring);
  printf("got %s, %d %d in entry \n", *ibufpp, len, **first_addr);

  disable_interrupts();
  //  put_sbuf_line (mystring, len, **first_addr);
  enable_interrupts();
}
//inline void indent(cons char **const ibufpp, int **first_addr, int ** second_addr){
////TODO: call indent library or system call to the current buffer.
//}

//TODO: format for indent command, try to find the header file for the library

//TODO: grep for all files in current directory recursively, this happens often enough it should be a succint command.
//TODO: //ask a chatbot via curl or local algorithm the given prompt, with context from first_addr to second_addr and insert the response into ibufpp at current address
//NOTE: this may require an additional address is current address in main_loop or buffer.c is not preserved after address passing
//TODO: ask(const char * ibufpp, int *first_addr, int *second_addr, char* prompt)

//TODO: //uses ctags c library to list all structs from first_addr to second_addr in ibufpp
//TODO: inline void structs(const char * ibufpp, int *first_addr, int *second_addr)

// This extension prints all currently registered marknos.
inline void
marknos(const char **const ibufpp, int **first_addr, int **second_addr)
{
  printf("Marknos: ");
  for (int i = 0; i < 26; ++i)
    {
      if (mark[i])
	{
	  printf("%c,  %i | ", i + 'a', get_line_node_addr(mark[i]));
	}
    }
  printf("\n");
}
