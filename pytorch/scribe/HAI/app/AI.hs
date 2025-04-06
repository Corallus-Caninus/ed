{-# LANGUAGE ForeignFunctionInterface #-}

{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeApplications #-}

module AI where

import           CPython.Types.Module (Module)
import           CPython.Simple (initialize, call, arg, PyCastException)
import           CPython.Simple.Instances () -- Import instances for using 'arg'
import           Data.Text (Text, pack)
import           Debug.Trace as Debug

-- | Run a Python command.
pyRun :: Text -> IO ()
pyRun cmd = Debug.trace ("pyRun: executing command: " ++ show cmd) $ do
  res <- call @() (pack "builtins") (pack "exec") [] [(pack "code", arg cmd)]
  Debug.trace ("pyRun: command executed: " ++ show cmd) $ return res
-- | The main function that runs the AI loop.

runAI :: IO ()
runAI = do
  initPython

-- | Initialize the Python interpreter (only once).
initPython :: IO ()
initPython = Debug.trace "initPython: initializing Python" $ do
  initialize
  Debug.trace "initPython: Python initialized"
initPython = do
  initialize
  -- Add the current directory to the Python path so it can find AI.py
  let cwd = "." -- Or use a more robust way to get the current working directory
  Debug.trace ("initPython: appending cwd to python path: " ++ cwd)
  pyRun $ pack "import sys"
  pyRun $ pack $ "sys.path.append('" ++ cwd ++ "')"
  Debug.trace "initPython: cwd appended to python path"
  -- Example: Call a Python function
