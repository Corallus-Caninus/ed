{-# LANGUAGE ForeignFunctionInterface #-}

{-# LANGUAGE LambdaCase #-}

module AI where

import           CPython.Types.Module (Module)

import           CPython.Simple (initialize, importModule, call, FromPy(fromPy), arg)
import           CPython.Simple.Instances () -- Import instances for using 'arg'
import           Data.Text (Text, pack)
import           System.IO.Unsafe (unsafePerformIO)

-- | Initialize the Python interpreter (only once).

-- | Call a Python function and return the result as a String.
callPythonFunction :: String -> String -> IO (Maybe String)
callPythonFunction moduleName functionName = do
  result <- importModule (pack moduleName) >>= \case
    Left err -> do -- Handle import error
      putStrLn $ "Error importing module: " ++ err
      return Nothing -- Return Nothing on error
    Right pyModule -> do -- Handle successful import
      result <- call pyModule (pack functionName) [] [] -- Call the Python function
      case result of -- Handle the result of the Python function call
        Right str -> return (Just str) -- Return the string result
        Left err -> do -- Handle Python function error
          putStrLn $ "Python function error: " ++ err
          return Nothing -- Return Nothing on error


initPython :: IO ()
initPython = do
  initialize
  -- Add the current directory to the Python path so it can find AI.py
  let cwd = "." -- Or use a more robust way to get the current working directory
  pyRun $ "import sys"
  pyRun $ "sys.path.append('" ++ cwd ++ "')" -- Use cwd as String

-- | Run a Python command.
pyRun :: String -> IO () -- Removed error handling
pyRun cmd = call (pack "builtins") (pack "exec") [] [(pack "code", arg cmd)] >> return ()
-- | The main function that runs the AI loop.
runAI :: IO ()
runAI = do
  initPython
  -- Example: Call a Python function
  result <- callPythonFunction "AI" "main_loop"
  case result of
    Just strResult -> do -- Handle Maybe String result
      putStrLn $ "Result from Python: " ++ strResult
    Nothing -> putStrLn "Failed to call Python function." -- Handle Nothing case
