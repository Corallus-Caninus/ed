module Main where

import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)
import           Debug.Trace as Debug
import           System.IO (hFlush, stdout)

main :: IO ()
main = do
  -- Import the AI module
  maybeAIModule <- importModule (pack "AI")
  case maybeAIModule of
    Just aiModule -> do
      putStrLn "AI module imported successfully."
      -- You can now call functions from the AI module using 'call'.
      -- Example:
      -- result <- call aiModule "some_function" [arg 1, arg "hello"]
      -- print result
    Nothing -> putStrLn "Failed to import AI module."

  putStrLn "Finished Main.main"
