module WeakPlayer where

import Data.Char (ord)
import Data.Ord
import Data.List
import Data.Function
import System.IO
import System.Time
import System.Timeout
import IsolaShared

-- When you program your tournament player, set this to your name ("firstname lastname")
getName = "Weak Player"

-- Returns the chosen move and its maximum search depth for a given state.
getMove :: State -> IO (Move, Int)
getMove state = timedDeepening state 1

-- Structure to hold a potential move, the board state after executing it, and the score for that state 
data MoveInfo = MoveInfo {move :: Move, newState :: State, newScore :: Double}

-- Receives the current state and the initial maximum search depth.
-- Returns the chosen move and its maximum search depth.
timedDeepening :: State -> Int -> IO (Move, Int)
timedDeepening state initMaxDepth =
    do startTime <- getClockTime
       deepen initMaxDepth nullMove startTime
       where deepen maxDepth bestMove sTime =
                 do currTime <- getClockTime
                    let diffTime = diffClockTimes currTime sTime
                    let remainTime = mcrsecPerMove - 1000000 * tdSec diffTime - (fromIntegral $ tdPicosec diffTime) `div` 1000000
                    compMove <- timeout remainTime (return $! findBestMove state 0 (-winScore - 1) (winScore + 1) maxDepth)                    
                    loop compMove
                    where loop Nothing  = return (bestMove, maxDepth - 1)
                          loop (Just (move, score))
                              | abs score > winScore = return (move, maxDepth)
                              | otherwise            = deepen (maxDepth + 1) move sTime

-- findBestMove receives the current state, the best score found so far, current alpha and beta, 
-- and the current number of levels that we are above maxDepth.
-- It uses Minimax and alpha-beta pruning to find the strongest move and return this move and its
-- associated score.
findBestMove :: State -> Double -> Double -> Double -> Int -> (Move, Double)
findBestMove st staticScore alpha beta level
    | level == 0 || abs staticScore > winScore = (nullMove, staticScore)
    | pl1Turn st = lookAheadMax nullMove alpha sMoves
    | otherwise  = lookAheadMin nullMove beta sMoves
                   where sMoves = sortedMoves st
                         lookAheadMax topMove topScore [] = (topMove, topScore)
                         lookAheadMax topMove topScore (m:ms)
                             | scAlpha >= beta     = (move m, scAlpha + 1)
                             | scAlpha >= topScore = lookAheadMax (move m) scAlpha ms
                             | otherwise           = lookAheadMax topMove topScore ms
                             where scAlpha = snd $ findBestMove (newState m) (newScore m) topScore beta  (level - 1) 
                         lookAheadMin topMove topScore [] = (topMove, topScore)
                         lookAheadMin topMove topScore (m:ms)
                             | scBeta <= alpha     = (move m, scBeta - 1)                              
                             | scBeta <= topScore  = lookAheadMin (move m) scBeta  ms
                             | otherwise           = lookAheadMin topMove topScore ms
                             where scBeta  = snd $ findBestMove (newState m) (newScore m) alpha topScore (level - 1) 

-- For a given state, returns a sorted MoveInfo list in which the most favorable moves appear first.
-- This sorting is used by findBestMove to make alpha-beta pruning as effective as possible.
sortedMoves :: State -> [MoveInfo]
sortedMoves st = if pl1Turn st then reverse sortedList else sortedList
    where sortedList = sortBy (comparing newScore) moveList 
          moveList = [MoveInfo {move = m, newState = nc, newScore = ns} 
                         | m <- genMoves st, let nc = execMove st m, let ns = score nc]
       
-- If the absolute score exceeds winScore, it means that the game is over (saves extra check) 
winScore = 999.0           

-- score is the e(p) function, where p is the state. Player 1 is Max, Player 2 is Min.
score :: State -> Double                                                          
score st
    | m1 == 0 && m2 == 0 && pl1Turn st = -1000.0
    | m1 == 0 && m2 == 0               =  1000.0
    | m1 == 0                          = -1000.0
    | m2 == 0                          =  1000.0
    | otherwise                        = fromIntegral (m1 - m2)
    where m1 = length $ genMoves st {pl1Turn = True}
          m2 = length $ genMoves st {pl1Turn = False}