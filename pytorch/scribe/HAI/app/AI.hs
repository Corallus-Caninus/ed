{-# LANGUAGE ForeignFunctionInterface #-}

{-# LANGUAGE LambdaCase #-}

module AI where

import           CPython.Types.Module (Module)

import           CPython.Simple (initialize, importModule, call, FromPy(fromPy), arg)
import           CPython.Simple.Instances () -- Import instances for using 'arg'
import           Data.Text (Text, pack)

-- | Initialize the Python interpreter (only once).
initPython :: IO ()
initPython = do
  initialize
  -- Add the current directory to the Python path so it can find AI.py
  let cwd = "." -- Or use a more robust way to get the current working directory
  pyRun $ "import sys"
  pyRun $ "sys.path.append('" ++ cwd ++ "')" -- Use cwd as String

-- | Run a Python command.
-- | Initialize the Python interpreter (only once).
-- | Initialize the Python interpreter (only once).
pyRun cmd = call (pack "builtins") (pack "exec") [] [(pack "code", arg cmd)] >> return ()
-- | The main function that runs the AI loop.
runAI :: IO ()
runAI = do
  initPython
