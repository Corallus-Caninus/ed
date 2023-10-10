#include <string.h>
//TODO: extract to config? this should be automated but if manually entered it should be extracted.
#define  NUM_COMMANDS 2


//TODO: one of the first commands should be a command that automatically generates and synchronizes the call table with the function prototypes
// COMMAND PROTOTPYES //
void test_print (int *, int *);
// END OF COMMAND PROTOTYPES //

/* CALL TABLE */
static const char *commands_s[] = { "test_print\n" };

//enum that reflects each command, these are ordered exactly as in commands_s
static const enum Command
{
  TEST_PRINT=1, // start at 1 so 0 is default in the jump table switch statement
};

//NOTE: ibufpp is newline terminated and can be assumed sanitized otherwise here since we are calling in switch statement of main_loop.c
int
parse_command (const char *ibufpp, int *first_addr, int *second_addr)
{
  printf ("%s %i, %i \n", ibufpp, *first_addr, *second_addr);	//TODO: remove after testing

   int command_index=0;
  for (int i=1; i<NUM_COMMANDS; i++)
    {
	if(strcmp(commands_s[i-1],ibufpp)==0){
	command_index=i;
	break;
	}
    }
   enum Command   command_select = (enum Command) command_index;

  switch (command_select)
    {
    case TEST_PRINT:
	test_print(&first_addr,&second_addr);
      break;
    default:
      printf ("command not found..\n");
    }
  return 0;			//TODO: error codes with breaks
}

void test_print(int *first_addr, int *second_addr){
	printf("inside test_print!\n");
}
