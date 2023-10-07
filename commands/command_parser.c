//TODO: print out the buffer to ensure pointer gets passed correctly and doesnt drop
//NOTE: ibufpp is newline terminated and can be assumed sanitized otherwise here since we are calling in switch statement of main_loop.c
void parse_commands(const char * ibufpp){
    printf("%s", ibufpp);
}
