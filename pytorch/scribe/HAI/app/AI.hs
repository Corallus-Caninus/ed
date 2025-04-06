{-# LANGUAGE ForeignFunctionInterface #-}

{-# LANGUAGE LambdaCase #-}
module AI where

import           CPython
import           CPython.Types (PyObject)
import           System.IO.Unsafe (unsafePerformIO)

-- | Initialize the Python interpreter (only once).
initPython :: IO ()
initPython = do
  initialize
  -- Add the current directory to the Python path so it can find AI.py
  let cwd = "." -- Or use a more robust way to get the current working directory
  pyRun "import sys"
  pyRun $ "sys.path.append('" ++ cwd ++ "')"

-- | Run a Python command.
pyRun :: String -> IO ()
pyRun cmd = do
  result <- pyExec cmd
  case result of
    Left err -> putStrLn $ "Python error: " ++ err
    Right _  -> return ()

-- | Call a Python function.
pyCall :: String -> String -> IO (Maybe PyObject)
pyCall moduleName functionName = do
  pyImport moduleName >>= \case
    Left err -> do
      putStrLn $ "Error importing module: " ++ err
      return Nothing
    Right pyModule -> do
      pyModule `pyCallMethod` functionName []
      return Nothing -- pyCallMethod returns IO (), we need to return IO (Maybe PyObject)
                   -- Returning Nothing here as a placeholder, you might want to handle the result properly

-- | Get a string from a PyObject.
pyObjectToString :: PyObject -> IO String
pyObjectToString obj = do
  result <- pyStr obj
  case result of
    Left err -> do
      putStrLn $ "Error converting to string: " ++ err
      return ""
    Right str -> return str

-- | The main function that runs the AI loop.
runAI :: IO ()
runAI = do
  initPython
  -- Example: Call a Python function
  result <- pyCall "AI" "main_loop"
  case result of
    Just pyObj -> do
      strResult <- pyObjectToString pyObj
      putStrLn $ "Result from Python: " ++ strResult
      pyDecref pyObj
    Nothing -> putStrLn "Failed to call Python function."
