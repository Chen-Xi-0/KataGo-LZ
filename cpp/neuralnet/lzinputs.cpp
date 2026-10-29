//V8 (LZ-compatible) input plane filling: implementation of NNInputs::fillRowV8.

#include "../neuralnet/nninputs.h"

#include <cassert>
#include <algorithm>

using namespace std;

//Local copy of the helper in nninputs.cpp.
static void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}

void NNInputs::fillRowV8(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  // LZ-compatible inputs (model version 18), 19x19 only - enforced by NNEvaluator.
  assert(nnXLen <= NNPos::MAX_BOARD_LEN);
  assert(nnYLen <= NNPos::MAX_BOARD_LEN);
  assert(board.x_size == nnXLen);
  assert(board.y_size == nnYLen);
  std::fill(rowBin,rowBin+NUM_FEATURES_SPATIAL_V8*nnXLen*nnYLen,false);
  std::fill(rowGlobal,rowGlobal+NUM_FEATURES_GLOBAL_V8,0.0f);

  int xSize = board.x_size;
  int ySize = board.y_size;

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NNInputs::NUM_FEATURES_SPATIAL_V8;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  Player pla = nextPlayer;
  Player opp = getOpp(pla);

  //Feature 0: on-board plane, doubles as the backend mask.
  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);
    }
  }

  //Features 1-8: stones of the player to move, h=0 (most recent)..7; 9-16: opponent.
  //Planes beyond the available history are zero-filled (LZ's gather_features).
  //KataGo y runs top-to-bottom while LZ planes index bottom-to-top, so board point (x,y)
  //is written at NN position (x, y_size-1-y); converted policy rows are y-flipped to match.
  int numMoves = std::min<int>((int)hist.moveHistory.size() + 1, 8);
  for(int h = 0; h<8; h++) {
    if(h >= numMoves)
      continue;
    const Board& b = hist.getRecentBoard(h);
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        int pos = NNPos::xyToPos(x, ySize - 1 - y, nnXLen);
        Loc loc = Location::getLoc(x,y,xSize);
        Color stone = b.colors[loc];
        if(stone == pla)
          setRowBin(rowBin,pos,1+h, 1.0f, posStride, featureStride);
        else if(stone == opp)
          setRowBin(rowBin,pos,9+h, 1.0f, posStride, featureStride);
      }
    }
  }

  //Features 17-18: LZ color planes. With dynamic komi, encode virtual komi K as
  //0.5 +/- K/15 so K=7.5 reproduces the usual 1/0.
  {
    float blackPlane = 1.0f;
    float whitePlane = 0.0f;
    if(nnInputParams.useDynamicKomi) {
      float plusKomi = 0.5f + nnInputParams.dynamicKomi / 15.0f;
      float minusKomi = 0.5f - nnInputParams.dynamicKomi / 15.0f;
      blackPlane = (nextPlayer == P_BLACK) ? plusKomi : minusKomi;
      whitePlane = (nextPlayer == P_BLACK) ? minusKomi : plusKomi;
    }
    else {
      blackPlane = (nextPlayer == P_BLACK) ? 1.0f : 0.0f;
      whitePlane = (nextPlayer == P_BLACK) ? 0.0f : 1.0f;
    }
    for(int y = 0; y<ySize; y++) {
      for(int x = 0; x<xSize; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        setRowBin(rowBin,pos,17, blackPlane, posStride, featureStride);
        setRowBin(rowBin,pos,18, whitePlane, posStride, featureStride);
      }
    }
  }
}
