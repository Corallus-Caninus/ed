{-# LANGUAGE ForeignFunctionInterface #-}

{-# LANGUAGE LambdaCase #-}

module AI where

import           CPython.Simple (initialize, importModule, call, FromPy(fromPy), arg)
import           CPython.Simple.Instances () -- Import instances for using 'arg'
import           Data.Text (Text, pack)
import           System.IO.Unsafe (unsafePerformIO)

-- | Initialize the Python interpreter (only once).
initPython :: IO ()
initPython = do
  initialize
  -- Add the current directory to the Python path so it can find AI.py
  let cwd = "." -- Or use a more robust way to get the current working directory
  pyRun $ "import sys"
  pyRun $ "sys.path.append('" ++ cwd ++ "')" -- Use cwd as String

-- | Run a Python command.
pyRun :: String -> IO ()
pyRun cmd = do
  result <- call (pack "builtins") (pack "exec") [] [(pack "code", arg cmd)]
  case result of
    Left err -> putStrLn $ "Python error: " ++ err
    Right _  -> return ()

-- | Call a Python function.
pyCall :: String -> String -> IO (Maybe String)
pyCall moduleName functionName = do
  moduleOrError <- importModule (pack moduleName)
  case moduleOrError :: Either String CPython.Types.Module.Module of
    Left err -> do
      putStrLn $ "Error importing module: " ++ err
      return Nothing
    Right pyModule -> do
      result <- call (pyModule :: CPython.Types.Module.Module) (pack functionName) [] []
      case result of
        Right str -> return (Just str)
        Left err -> do
          putStrLn $ "Python function error: " ++ err
          return Nothing

-- | The main function that runs the AI loop.
runAI :: IO ()
runAI = do
  initPython
  -- Example: Call a Python function
  result <- pyCall "AI" "main_loop"
  case result of
    Just strResult -> do -- Handle Maybe String result
      putStrLn $ "Result from Python: " ++ strResult
    Nothing -> putStrLn "Failed to call Python function." -- Handle Nothing case
