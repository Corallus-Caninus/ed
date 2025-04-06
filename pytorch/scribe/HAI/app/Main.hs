module Main where

import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)
import           Debug.Trace as Debug
import           System.IO (hFlush, stdout)

main :: IO ()
main = do
  --Debug.trace "Main: Starting main function..." $ return ()
  --AI.runAI
  --Debug.trace "Main: AI.runAI call finished." $ return ()
  putStrLn "Starting Main.main"
  hFlush stdout
  aiMod <- importModule (pack "AI")
  case aiMod of
    Just ai_module -> do
      putStrLn "AI module imported successfully."
      runAIResult <- call ai_module (pack "runAI") []
      case runAIResult of
        Just result -> do
          putStrLn "AI.runAI call finished successfully."
          -- You might want to do something with the result here
        Nothing -> putStrLn "Error calling AI.runAI."
    Nothing -> putStrLn "Failed to import AI module."
  putStrLn "Finished Main.main"
