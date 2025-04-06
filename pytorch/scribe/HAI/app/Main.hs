module Main where

import           AI
import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)

main :: IO ()
main = do
  initPython
  -- Example: Call a Python function
