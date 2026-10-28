//LZ-compatible (model version 18) input plane filling, split out of neuralnet/nninputs.cpp so
//the upstream file keeps only a one-line dispatcher (see 05_验证输出/LZ分支代码解耦重构设计_20261028.md §4).
//The implementation stays as NNInputs::fillRowV8 - the declaration lives in nninputs.h next to
//the other input versions and every call site (nneval, trainingwrite) is unchanged.

#include "../neuralnet/nninputs.h"

#include <cassert>
#include <algorithm>

using namespace std;

//File-local helper duplicated from nninputs.cpp (the V8 fill is the only lzinputs code that
//needs it; keeping both copies static avoids touching upstream nninputs.cpp further).
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

  //Feature 0 - on board. This plane doubles as the backend mask (Model::apply takes
  //mask = input channel 0), so it must be exactly 1.0 on board and 0.0 off board.
  for(int y = 0; y<ySize; y++) {
    for(int x = 0; x<xSize; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      setRowBin(rowBin,pos,0, 1.0f, posStride, featureStride);
    }
  }

  //Features 1-8: the player to move's stones, h = 0 (most recent) .. 7
  //Features 9-16: the opponent's stones, h = 0 .. 7
  //LZ semantics (gather_features): moves = min(movenum+1, 8), and planes for h >= moves
  //stay zero. KataGo's recentBoards ring returns copies of the *starting board* beyond
  //the available history, which differs from LZ's zero-fill whenever the start position
  //has setup stones, so explicitly zero-fill instead of relying on getRecentBoard.
  //
  //Coordinate direction: KataGo's board y runs top-to-bottom (GTP row r <-> internal
  //y = y_size - r, see Location::tryOfString / Board::printBoard), while the
  //Leela-Zero plane layout indexes positions bottom-to-top: idx = y_b * 19 + x with
  //y_b = GTP row - 1. Writing each board point (x,y) at NN position (x, y_size-1-y)
  //makes the tensor consumed by the backend equal the LZ input planes exactly.
  //(The policy head's pFullMul output rows are correspondingly y-flipped in the
  //converted model file, so logits land on the correct GTP vertices.)
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

  //Features 17,18 - color planes, matching LZ's gather_features: LZ plane 16 is the
  //all-ones plane iff black is to move, LZ plane 17 iff white is to move.
  //With dynamic komi, follow LZ's "dynamic color planes": encode a virtual komi K as
  //plus = 0.5 + K/15 on the plane of the side that gives komi and minus = 0.5 - K/15
  //on the other, so K = 7.5 reproduces the usual 1/0 and K = 0 means no komi.
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
