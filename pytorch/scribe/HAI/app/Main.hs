module Main where

import           AI
import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)

main :: IO ()
main = do
  initPython
  -- Example: Call a Python function
  eitherPyModule <- importModule (pack "AI")
  case eitherPyModule of
    Left err -> do
      putStrLn $ "Error importing module: " ++ err
    Right pyModule -> do
      eitherResult <- call pyModule (pack "main_loop") [] []
      case eitherResult of
        Right strResult -> do -- Handle Maybe String result
          putStrLn $ "Result from Python: " ++ strResult
        Left err -> putStrLn $ "Python function error: " ++ err
    Nothing -> putStrLn "Failed to call Python function." -- Handle Nothing case

main :: IO ()
main = do
  initPython
  -- Example: Call a Python function
