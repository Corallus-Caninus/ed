#!/usr/bin/env bash
GROQ_API_KEY=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 GEMINI_API_KEY=AIzaSyBOfDsWNXLa0RbcpNGUhoOWSU8EPqyFwpA LD_LIBRARY_PATH=/nix/store/qksd2mz9f5iasbsh398akdb58fx9kx6d-gcc-13.2.0-lib/lib/ aider --model gemini/gemini-1.5-flash-latest --edit-format diff --dark-mode 
