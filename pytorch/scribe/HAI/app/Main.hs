module Main where

import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)
import           Debug.Trace as Debug
import           System.IO (hFlush, stdout)

main :: IO ()
main = do
  maybeAIModule <- importModule (pack "AI")
  case maybeAIModule of
    Just (aiModule :: Module) -> do {
        putStrLn "AI module imported successfully.";
        result <- call aiModule (pack "closure") [] ;
        print result
      }
    Nothing -> putStrLn "Failed to import AI module."

  putStrLn "Finished Main.main"
