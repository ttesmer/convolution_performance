{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE QuasiQuotes     #-}
{-# LANGUAGE TemplateHaskell #-}
module Main where

import           Codec.Compression.GZip       (decompress)
import           Control.Monad
import           Data.Array.Repa              as R
import           Data.Array.Repa.Stencil      as Stencil
import           Data.Array.Repa.Stencil.Dim2 as Stencil
import qualified Data.Massiv.Array            as MA
import qualified Data.ByteString              as BS
import qualified Data.ByteString.Lazy         as BL
import           Data.List                    as L
import qualified Numeric.LinearAlgebra        as LA
import           Prelude                      as P
import           System.Directory             (getCurrentDirectory)
import           System.Environment
import           System.IO
import           System.Random                as Rand
import           Control.DeepSeq              as DS

import           Foreign.Ptr                  ( Ptr )
import           System.IO.Unsafe             (unsafePerformIO)
import qualified Data.Vector.Storable as U    ( unsafeToForeignPtr0, unsafeFromForeignPtr0 )
import qualified Numeric.LinearAlgebra.Devel as U
import           Foreign                      ( mallocForeignPtrArray, withForeignPtr )

main :: IO ()
main = do
    args <- getArgs

    imgs <- loadData
    kernels <- initKernels
    case args of
      "--hmatrix":_ -> do
          runHmatrix imgs kernels
      "--massiv":_ -> do
          runMassiv imgs kernels
      "--repa":_ -> do
          runRepa imgs kernels
      "--cublas":_ -> do
          runCublas
      "--cuda":_ -> do
          runCuda
      _ -> putStrLn "Error"

runMassiv :: a -> b -> IO ()
runMassiv a b = putStrLn "TODO"

-- time:   1.54s user
-- space: ~2GB?
runHmatrix :: [LA.Matrix Double] -> [LA.Matrix Double] -> IO ()
runHmatrix imgs kernels =
    let kcols = LA.fromRows $ P.map LA.flatten kernels
        icols = (LA.tr . im2col 5 5 1 1) <$> imgs
        res = P.map (kcols LA.<>) icols
     in res `deepseq` putStrLn $ "Done, size: " P.++ (show . length $ res)
   
    --let dCircKernel = doublyBlockedCirculant . zeroPad (28, 28) . flipKernel $ k
    --sequence_ $
    --    (LA.dispShort 10 10 3) <$> [dCircKernel LA.<> img | img <- imgs]

-- time:   47.93s user
-- space: ~5GB
runRepa :: [LA.Matrix Double] -> [LA.Matrix Double] -> IO ()
runRepa imgs kernels = do
    let ks = hmatrix2repa <$> kernels
    let is = hmatrix2repa <$> imgs

    let imgMat = head is
    let kernel = [stencil2| 10 10 10 10 10
                            10 10 10 10 10
                            10 10 10 10 10
                            10 10 10 10 10
                            10 10 10 10 10 |] :: Stencil.Stencil DIM2 Double
    res <- forM is $ \i ->
        forM (replicate 20 kernel) $ \k ->
            computeP ((mapStencil2 (BoundConst 0) k i) :: Array PC5 DIM2 Double)
    print (res :: [[Array U DIM2 Double]])

runCublas = print ""

runCuda = print ""

--convolveP :: Array U DIM2 Double  -- ^ 28x28    Input image       (Unboxed Vector)
--          -> Array C DIM3 Double  -- ^ 20x5x5   Kernel tensor     (Cursored Array)
--          -> Array U DIM3 Double  -- ^ 20x24x24 Convolution layer (Unboxed Vector)

{- Kernel stuff -}
initKernels :: IO [LA.Matrix Double]
initKernels = replicateM 20 $ LA.randn 5 5

flipKernel :: LA.Matrix Double -> LA.Matrix Double
flipKernel = LA.flipud

doublyBlockedCirculant :: LA.Matrix Double  -- ^ input kernel
                       -> LA.Matrix Double
doublyBlockedCirculant m = LA.fromBlocks . circ $ circOfRows m

circOfRows :: LA.Matrix Double -> [LA.Matrix Double]
circOfRows m = [ LA.fromLists $ circ row | row <- rows ]
    where rows = reverse $ LA.toLists m

circ :: [a] -> [[a]]
circ xs = take (length xs) $ iterate (rotate 1) xs -- `(length xs)` is wrong here..

rotate :: Int -> [a] -> [a]
rotate n xs = drop diff xs <> take diff xs
    where diff = length xs - n

zeroPad :: (Int, Int) -> LA.Matrix Double -> LA.Matrix Double
zeroPad (r,c) m = LA.fromBlocks $
    [[LA.konst 0 (r-r', 1), LA.row $ replicate (c-c') 0],
     [m, LA.row $ replicate (c-c') 0]]
         where (r', c') = LA.size m

zeroPadTo :: Int -> Int -> [LA.Matrix Double] -> [LA.Matrix Double]
zeroPadTo r c kernels = [ LA.fromBlocks [[LA.konst 0 (r', 1), LA.row $ replicate r' 0], [k, LA.row $ replicate r' 0]] | k <- kernels]
    where (r', c') = (r-5, c-5)

getSizeZP :: Int -> Int -> (Int, Int)
getSizeZP m n = ((n-m+1)^2, n^2)

{- Load data -}
loadData :: IO [LA.Matrix Double]
loadData = do
    trainImgs <- getData "train-images-idx3-ubyte.gz"
    let imgs = BL.toStrict trainImgs
    let trainData = chunkList 10 [getImage n imgs | n <- [0..49999]]
    return $ LA.fromColumns <$> trainData

getImage :: Int -> BS.ByteString -> LA.Vector Double
getImage n imgs = LA.fromList [normalize $ BS.index imgs (16 + n*784 + s) | s <- [0..783]]

getData :: FilePath -> IO BL.ByteString
getData path = do
    currentDir <- getCurrentDirectory
    fileData <- decompress <$> BL.readFile (currentDir P.<> "/mnist_dataset/" P.<> path)
    return fileData

chunkList :: Int -> [a] -> [[a]]
chunkList n xs = takeWhile (not.null) $ unfoldr (Just . splitAt n) xs

normalize :: (Integral a, Floating b) => a -> b
normalize x = fromIntegral x/255

hmatrix2repa :: LA.Matrix Double -> R.Array R.U R.DIM2 Double
hmatrix2repa mat = R.fromListUnboxed (R.Z R.:. (r::Int) R.:. (c::Int)) . LA.toList . LA.flatten $ mat
    where (r, c) = LA.size mat

foreign import ccall unsafe im2col_cpu :: Ptr Double -> Int -> Int -> Int -> Int -> Int -> Int -> Int -> Ptr Double -> IO ()

im2col_c :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> LA.Matrix Double -> LA.Matrix Double
im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataImg = 
    let vec = LA.flatten dataImg
        rowOut = (height - kernelRows) `div` strideRows + 1
        colOut = (width - kernelColumns) `div` strideColumns + 1
        kernelSize = kernelRows * kernelColumns
        numberOfPatches = rowOut * colOut
    in unsafePerformIO $ do
        outPtr <- mallocForeignPtrArray (numberOfPatches * kernelSize * channels)
        let (inPtr, _) = U.unsafeToForeignPtr0 vec
    
        withForeignPtr inPtr $ \inPtr' ->
            withForeignPtr outPtr $ \outPtr' ->
                im2col_cpu inPtr' channels height width kernelRows kernelColumns strideRows strideColumns outPtr'

        let matVec = U.unsafeFromForeignPtr0 outPtr (numberOfPatches * kernelSize * channels)
        return $ U.matrixFromVector U.RowMajor numberOfPatches (kernelSize * channels) matVec

im2col :: Int -> Int -> Int -> Int -> LA.Matrix Double -> LA.Matrix Double
im2col kernelRows kernelColumns strideRows strideColumns dataIm =
  let channels = 1
      height = LA.rows dataIm
      width  = LA.cols dataIm
  in  im2col_c channels height width kernelRows kernelColumns strideRows strideColumns dataIm

