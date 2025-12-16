import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaConfig, Mamba2ForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model

#print(f"Number of CUDA devices available: {torch.cuda.device_count()}")

#model_id = "mistralai/Mamba-Codestral-7B-v0.1" # Or your desired base model ID
#model_id = "mistralai/Mamba-Codestral-7B-v0.1" # Or your desired base model ID
model_id = "AntonV/mamba2-370m-hf"
filename = "AI_Checkpoint.ai" # Path to your LoRa adapter checkpoint

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

if os.path.exists(filename): # Load model weights and LoRa adapter
    print(f"Checkpoint file '{filename}' found. Loading LoRa adapter from checkpoint...")
    config = MambaConfig.from_pretrained(filename, trust_remote_code=True, vocab_size=50288) # Load base config, force vocab_size to match checkpoint
    model = Mamba2ForCausalLM.from_pretrained(filename, config=config,  torch_dtype=torch.float16, ignore_mismatched_sizes=True, device_map="cpu", trust_remote_code=True)
#    lora_config = LoraConfig.from_pretrained(filename) # Load LoRa config from checkpoint
#    model = PeftModel.from_pretrained(model, filename) # Load LoRa weights
    print(f"LoRa adapter loaded successfully from '{filename}'.")
else:
    print(f"Checkpoint file '{filename}' not found. Please ensure the LoRa adapter checkpoint exists.")
    exit()

model = model#.to(dtype#.torch.float16)#.to("cuda")
model.eval() # Set model to evaluation mode

#prompt = "-- A Haskell Module that opens a file and prints it to stdout:" # Or your desired prompt
prompt = '''
{-# LANGUAGE LambdaCase #-}
-- | This module encapsulates all logic related to interpreting and applying
-- Address types to the editor's buffer content. It provides a clean abstraction
-- layer for the Commands module, reducing its complexity.
module AddressLogic
(
 getLinesByAddress,
 writeLines,
 deleteLines,
 copyLines,
 editLines
)
where

import Types (Address(..), RegexPatternComponent(..), SubPatternSource(..), CommandResult(..), rhoCompOpt)
import Data.List (sortBy, nub, find)
import Data.Maybe (fromMaybe)
import Text.Regex.TDFA (Regex, matchTest, makeRegexOpts, defaultExecOpt)
import Utils (subRegexGlobal)
import qualified Data.Map as Map
import qualified Data.Set as Set
import Control.Exception (try, evaluate, SomeException, displayException)
import System.IO.Unsafe (unsafePerformIO)

-- | Matches a single compiled 'Regex' against a line.
matchSubPattern :: Regex -> String -> Bool
matchSubPattern = matchTest

-- | Safely compiles a regex pattern.
-- If compilation fails, it prints the error
-- to stdout and returns a Left error message.
-- Otherwise, returns Right Regex.
-- Uses unsafePerformIO to bridge IO (for exception handling and printing) and pure code.
safeCompileRegex :: String -> Either String Regex
safeCompileRegex pattern = unsafePerformIO $ do
    result <- try (evaluate (makeRegexOpts rhoCompOpt defaultExecOpt pattern)) :: IO (Either SomeException Regex)
    case result of
        Left e -> do
            let errMsg = "Regex compilation error for '" ++ pattern ++ "': " ++ displayException e
            putStrLn errMsg
            return $ Left errMsg
        Right regex -> return $ Right regex

-- | Resolves a 'SubPatternSource' into a list of compiled 'Regex' values and their
-- original pattern strings. This function handles 'LiteralPattern' and recursively
-- resolves 'PatternRefByName'.
resolveSubPatternSource :: Map.Map String Address -> SubPatternSource -> Either String [(String, Regex)]
resolveSubPatternSource savedAddressMap = \case
    LiteralPattern s ->
        if null s
            then Right []
            else case safeCompileRegex s of
                     Left err -> Left err
                     Right regex -> Right [(s, regex)]
    PatternRefByName name ->
        case Map.lookup name savedAddressMap of
            Just (RegexAddress rpc) -> fmap concat $ traverse (resolveSubPatternSource savedAddressMap) (subPatternSources rpc)
            Just _ -> Left ("Error: Saved address '" ++ name ++ "' is not a RegexAddress.")
            Nothing -> Left ("Error: Saved address '" ++ name ++ "' not found.")

-- | Checks if a line matches all sub-patterns within a 'RegexPatternComponent'.
-- It uses 'resolveSubPatternSource' to handle 'PatternRefByName's and compile regexes.
matchAllSubPatterns :: Map.Map String Address -> RegexPatternComponent -> String -> Bool
matchAllSubPatterns savedAddressMap rpc line =
    if null (subPatternSources rpc)
        then True
        else case fmap concat $ traverse (resolveSubPatternSource savedAddressMap) (subPatternSources rpc) of
                 Left _err -> False
                 Right pairedRegexes -> all (\(_, regex) -> matchSubPattern regex line) pairedRegexes

-- | Sorts and removes duplicates from a list of integers in ascending order.
uniqueSortAsc :: [Int] -> [Int]
uniqueSortAsc = sortBy compare . nub

-- | Inserts a list of items into another list at a specific 0-based index.
insertListAt :: Int -> [a] -> [a] -> [a]
insertListAt idx newItems xs = take idx xs ++ newItems ++ drop idx xs

-- | Deletes lines from a list by their 1-based indices.
deleteByIndices :: [Int] -> [a] -> [a]
'''
input_ids = tokenizer(prompt, return_tensors="pt").input_ids#.to("cuda")

#print(f"--- Before generate - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#print(f"--- Before generate - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

with torch.no_grad():
    generated_ids = model.generate(input_ids, max_length=2000, num_return_sequences=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

#print(f"--- After generate - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
#print(f"--- After generate - CUDA memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

print("\nGenerated Text:")
print(generated_text)
