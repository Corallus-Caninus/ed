//#include "../buffer.c"
//TODO: print out the buffer to ensure pointer gets passed correctly and doesnt drop
//NOTE: ibufpp is newline terminated and can be assumed sanitized otherwise here since we are calling in switch statement of main_loop.c
void parse_commands(const char * ibufpp, int first_addr, int second_addr){
    printf("%s, %i, %i",  ibufpp, first_addr, second_addr);
//    printf("%i", current_addr_);
// TODO: pass in proper address from main_loop since this is properly lexd there
}
