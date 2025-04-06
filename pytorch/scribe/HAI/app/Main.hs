module Main where

import           CPython.Types.Module (Module)
import           CPython.Simple (importModule, call, FromPy(fromPy), arg)
import           Data.Text (Text, pack)
import           Debug.Trace as Debug

main :: IO ()
main = do
  --Debug.trace "Main: Starting main function..." $ return ()
  --AI.runAI
  --Debug.trace "Main: AI.runAI call finished." $ return ()
  putStrLn "Starting Main.main"
--  runAI
  putStrLn "Finished Main.main"
