module Main where

import           AI
import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)

main :: IO ()
main = do
  initPython
  -- Example: Call a Python function
  result <- callPythonFunction "AI" "main_loop"
  case result of
    Just strResult -> do -- Handle Maybe String result
      putStrLn $ "Result from Python: " ++ strResult
    Nothing -> putStrLn "Failed to call Python function." -- Handle Nothing case
  where
    -- | Call a Python function and return the result as a String.
    callPythonFunction :: String -> String -> IO (Maybe String)
    callPythonFunction moduleName functionName = do
      eitherPyModule <- importModule (pack moduleName)
      case eitherPyModule of
        Left err -> do
          putStrLn $ "Error importing module: " ++ err
          return Nothing
        Right pyModule -> do
          eitherResult <- call pyModule (pack functionName) [] []
          case eitherResult of
            Right str -> return $ Just str
            Left err -> do
              putStrLn $ "Python function error: " ++ err
              return Nothing
