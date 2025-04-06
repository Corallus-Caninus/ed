module Main where

import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)
import           Debug.Trace as Debug
import           System.IO (hFlush, stdout)

main :: IO ()
main = do
  putStrLn "Finished Main.main"
