#include <string.h>
//TODO: calling this extensions is a bit confusing given the info file. consider "mods" or "extensions"
//TODO: extract to config? this should be automated but if manually entered it should be extracted.
#define  NUM_EXTENSIONS 1+1 //TODO: is this correct? could this be 1?

//extern void get_filename (const char **const ibufpp);

/* EXTENSION PROTOTPYES */
void test_print (int *, int *);
void funcs(const char **, const char*, int*, int*);
// END OF EXTENSION PROTOTYPES //

/* CALL TABLE */
static const char *extensions_s[] =
{
	"test_print\n",
	"funcs\n",
};

/* JUMP TABLE EXTENSION ENUMERATION */
// These are ordered exactly as in extensions_s
static const enum Extension
{
  TEST_PRINT=1, // start at 1 so 0 is default "not found" in switch statement
  FUNCS,
};

/* PARSER */
//TODO: ibufpp might be pass by value here, if so just pass by reference
//Extensions should have argument format: inline int EXTENSION_NAME(const char ** const ibufpp, int *first_addr, int *second_addr, char* args)
//TODO: pass in filename here
int parse_extension (const char ** const ibufpp, const char* filename, int *first_addr, int *second_addr)
{
  printf ("%s %i, %i \n", ibufpp, *first_addr, *second_addr);	//TODO: remove after testing

  //find the extension enum from the call table given the ibufpp 
  int extension_index=0;
  enum Extension extension_select;
  for (int i=1; i<NUM_EXTENSIONS; i++)
    {
	if(strcmp(extensions_s[i-1],*ibufpp)==0){
		extension_index=i;
	break;
	}
    }
    extension_select = (enum Extension) extension_index;

  //call the extension given the selection, this can be used for error handling and data preperation as in main_loop.c enum
  switch (extension_select)
    {
    case TEST_PRINT:
	test_print(&first_addr,&second_addr);
         break;
    case FUNCS:
	funcs(&ibufpp, &filename, &first_addr, &second_addr);
        break;
    default:
      printf ("extension not found..\n");
    }
  return 0;			//TODO: error codes with breaks
}

/* BEGIN EXTENSION DECLARATIONS */

inline void test_print(int *first_addr, int *second_addr){
	printf("inside test_print!\n");
}

//uses ctags c library to list all funcs from first_addr to second_addr in ibufpp
inline void funcs(const char ** const ibufpp, const char *filename, int *first_addr, int *second_addr){
   //char* ctags_extension = calloc(100);
//     const char* filename = get_filename(ibufpp);
    printf(filename);
//TODO: calloc needs to be static bytes for ctags -x + dynamic filename
//   strcat(ctags_extension, "ctags -x ");
   //ctags_extension -x $cur_filename
}

//TODO: //ask a chatbot via curl or local algorithm the given prompt, with context from first_addr to second_addr and insert the response into ibufpp at current address
//NOTE: this may require an additional address is current address in main_loop or buffer.c is not preserved after address passing
//TODO: ask(const char * ibufpp, int *first_addr, int *second_addr, char* prompt)

//TODO: //uses ctags c library to list all structs from first_addr to second_addr in ibufpp
//TODO: inline void structs(const char * ibufpp, int *first_addr, int *second_addr)

//TODO: //uses ctags c library to list all variables from first_addr to second_addr in ibufpp including constants and statics
//TODO: inline void variables(const char * ibufpp, int *first_addr, int *second_addr)
