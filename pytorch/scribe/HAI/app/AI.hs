{-# LANGUAGE ForeignFunctionInterface #-}

{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE ScopedTypeVariables #-}

module AI where

import           CPython.Types.Module (Module)
import           CPython.Simple (initialize, call, arg, PyCastException, importModule)
import           CPython.Simple.Instances () -- Import instances for using 'arg'
import           CPython.Types (PyObject)
import qualified CPython.Simple as Simple
import           Debug.Trace as Debug
import           Control.Exception (catch, SomeException)
import           Data.Text (Text, pack)


runAI :: IO ()
runAI = do
  Debug.trace "runAI: Initializing CPython..." $ return ()
  initialize
  Debug.trace "runAI: CPython initialized." $ return ()
  putStrLn "CPython initialized."
  Debug.trace "runAI: Importing AI module..." $ return ()
  aiModule <- importModule (pack "add") `catch` \(e :: SomeException) -> (do
    Debug.trace "runAI: SomeException caught!" $ return ()
    putStrLn "SomeException caught!"
    putStrLn $ "Error message: " ++ show e
    error "Failed to import AI module due to SomeException"
    )
  Debug.trace "runAI: AI.py module imported." $ return ()
  putStrLn "AI.py module imported."
  putStrLn "Running AI.py..."
  Debug.trace "runAI: Executing AI.py top-level code..." $ return ()
  -- No need to explicitly call a main function in AI.py for simple execution
  result <- call aiModule (pack "add_numbers") [arg @Int 5, arg @Int 3]
  case result of
    Just value -> do
      putStrLn $ "Result from Python: " ++ show value
    Nothing -> putStrLn "Failed to get result from Python."
  putStrLn "AI.py execution finished (if it had top-level code)."
  Debug.trace "runAI: AI.py execution finished." $ return ()
  return ()
