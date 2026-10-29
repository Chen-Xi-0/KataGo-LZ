#include "../neuralnet/lzladder.h"

#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../core/test.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

//LZ ladder correction: V8 inputs have no ladder planes, so policy and value are corrected
//from solver verdicts (policy surgery) and from evaluating the resolved capture (value blend).
//LZ models only.

namespace LzLadder {
  Params params;
  std::atomic<int> rootArmed(0);
  Params::Params()
    : enabled(true), minStones(1),
      checkNodes(10), minPolicy(5e-4), policyFactor(0.9), penaltyRank(2), chaseStones(2),
      boostFactor(2.0), defenseDepth(11), offenseEnabled(true), offenseDepth(6),
      boostWinrate(0.3), penaltyWinrate(0.9), minDepth(6), ladderValueBlend(0.8) {}
}

//Shared predicate used by the boost, the root arming scan and the value blend.
namespace LzLadder {
  //First empty liberty of the chain. A one-liberty group has exactly one by definition.
  static Loc findChainLiberty(const Board& board, Loc loc) {
    Loc lib = Board::NULL_LOC;
    Loc head = board.chain_head[loc];
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        if(board.chain_head[Location::getLoc(x,y,board.x_size)] != head)
          continue;
        for(int d = 0; d < 4; d++) {
          const int nx = x + (d==0?1:d==1?-1:0);
          const int ny = y + (d==2?1:d==3?-1:0);
          if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= board.y_size)
            continue;
          Loc nb = Location::getLoc(nx,ny,board.x_size);
          if(board.colors[nb] == C_EMPTY)
            return nb;
        }
      }
    }
    return lib;
  }

  bool isRealLadderEscape(
    const Board& board, const BoardHistory& history, Loc loc, Player pla,
    int minStones, int minDepth, Loc* outMove, int* outDepth
  ) {
    if(outMove != NULL) *outMove = Board::NULL_LOC;
    if(outDepth != NULL) *outDepth = 0;
    if(loc < 0 || loc >= Board::MAX_ARR_SIZE || board.colors[loc] != pla)
      return false;
    if(board.chain_head[loc] != loc)
      return false;
    if(board.getNumLiberties(loc) != 1 || board.getChainSize(loc) < minStones)
      return false;
    Loc lib = findChainLiberty(board, loc);
    if(lib == Board::NULL_LOC)
      return false;
    Board trial(board);
    BoardHistory trialHist(history);
    if(!trialHist.makeBoardMoveTolerant(trial,lib,pla))
      return false;
    if(trial.ko_loc != Board::NULL_LOC)
      return false;
    if(trial.getNumLiberties(lib) != 2)
      return false;
    vector<Loc> buf;
    int d = 0;
    if(trial.searchIsLadderCaptured(lib,false,buf,&d))
      return false;
    //Depth 0 means the node budget ran out; only long ladders need the value correction.
    if(d < minDepth)
      return false;
    if(outMove != NULL) *outMove = lib;
    if(outDepth != NULL) *outDepth = d;
    return true;
  }

  bool isEscapedTwoLibertyGroup(
    const Board& board, Loc loc, Player pla, int minStones, int minDepth,
    Loc* outMove, int* outDepth
  ) {
    if(outMove != NULL) *outMove = Board::NULL_LOC;
    if(outDepth != NULL) *outDepth = 0;
    if(loc < 0 || loc >= Board::MAX_ARR_SIZE || board.colors[loc] != pla)
      return false;
    if(board.chain_head[loc] != loc)
      return false;
    if(board.getNumLiberties(loc) != 2 || board.getChainSize(loc) < minStones)
      return false;
    vector<Loc> buf;
    int d = 0;
    Board probe(board);
    if(probe.searchIsLadderCaptured(loc,false,buf,&d))
      return false;
    if(d < minDepth)
      return false;
    if(outMove != NULL) *outMove = findChainLiberty(board, loc);
    if(outDepth != NULL) *outDepth = d;
    return true;
  }

  //Liberties after each move, replayed from the initial board (only the last 8 are kept).
  static std::vector<int> computeLadderLibsAfter(const BoardHistory& history) {
    const std::vector<Move>& mh = history.moveHistory;
    const int m = (int)mh.size();
    std::vector<int> libsAfter((size_t)m, 0);
    Board replay(history.initialBoard);
    for(int i = 0; i < m; i++) {
      const Move& mv = mh[i];
      if(mv.loc <= Board::PASS_LOC)
        continue;
      replay.playMoveAssumeLegal(mv.loc, mv.pla);
      if(replay.colors[mv.loc] == C_EMPTY)
        continue;
      libsAfter[i] = replay.getNumLiberties(mv.loc);
    }
    return libsAfter;
  }

  //Arms the value correction if the root has a long-ladder escape for `pla`.
  void armRootScan(
    const Board& board, const BoardHistory& hist, Player pla, std::atomic<int>& armedOut
  ) {
    int armed = 0;
    if(params.enabled && board.ko_loc == Board::NULL_LOC) {
      for(int y = 0; y < board.y_size && armed == 0; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          Loc move = Board::NULL_LOC;
          int d = 0;
          if(isRealLadderEscape(
               board, hist, loc, pla, params.minStones, params.minDepth,
               &move, &d)) {
            if(getenv("LZ_LADDER_DEBUG") != NULL)
              std::fprintf(stderr, "[LADR] armed at %s d=%d\n", Location::toString(move,board).c_str(), d);
            armed = 1;
            break;
          }
        }
      }
    }
    armedOut.store(armed, std::memory_order_relaxed);
  }

  //Counts chased stones (walking backwards) that had two liberties when played.
  static int countLadderHistory(
    const Board& curBoard, const BoardHistory& history, const std::vector<int>& libsAfter, Loc chasedLoc
  ) {
    const std::vector<Move>& mh = history.moveHistory;
    const int m = (int)mh.size();
    if(m <= 0 || (int)libsAfter.size() != m)
      return 0;
    int counter = 0;
    Loc currentMove = chasedLoc;
    for(int idx = m - 1; idx >= 0; idx -= 2) {
      const Move& mv = mh[idx];
      if(mv.loc <= Board::PASS_LOC)
        break;
      if(mv.loc == chasedLoc)
        continue;
      if(curBoard.chain_head[mv.loc] != curBoard.chain_head[currentMove])
        continue;
      if(libsAfter[idx] == 2) {
        counter++;
        currentMove = mv.loc;
      }
      else
        break;
    }
    return counter;
  }
}

void applyLZLadderCorrectionV5(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  int nnXLen,
  int nnYLen,
  float* policy,
  const bool* isLegal,
  float* winMulOut
) {
  const LzLadder::Params& P = LzLadder::params;
  const int policySize = NNPos::getPolicySize(nnXLen,nnYLen);

  float maxLogit = -1e25f;
  for(int pos = 0; pos<policySize; pos++) {
    if(!isLegal[pos])
      continue;
    if(policy[pos] > maxLogit)
      maxLogit = policy[pos];
  }
  if(maxLogit <= -1e24f)
    return;
  //Ko makes the ladder verdict unreliable; skip the surgery.
  if(board.ko_loc != Board::NULL_LOC)
    return;
  //Original softmax denominator over legal moves, used by the winrate nudges.
  float sumExp = 0.0f;
  for(int pos = 0; pos<policySize; pos++) {
    if(isLegal[pos])
      sumExp += std::exp(policy[pos] - maxLogit);
  }

  const bool suicideLegal = history.rules.multiStoneSuicideLegal;
  static const int NDX[4] = {1,-1,0,0};
  static const int NDY[4] = {0,0,1,-1};
  vector<Loc> buf;
  vector<Loc> workingMoves;

  //Liberty table shared by all candidates of this eval.
  vector<int> ladderLibsAfter;
  bool ladderLibsComputed = false;
  const auto getLadderCounter = [&](const Board& curBoard, Loc chasedLoc) {
    if(!ladderLibsComputed) {
      ladderLibsAfter = LzLadder::computeLadderLibsAfter(history);
      ladderLibsComputed = true;
    }
    return LzLadder::countLadderHistory(curBoard, history, ladderLibsAfter, chasedLoc);
  };

  //boost : move puts an opponent group in atari that cannot escape -> lift the prior.
  //demote: move leaves an own two-liberty group that is ladder-captured -> lower the prior.
  //chase : atari on a group that escapes; deep enough into a futile chase -> zero the move.
  //---------------------------------------------------------------------------
  //Group-driven rescue boost: the candidate pass below only looks at the top priors, so walk
  //own one-liberty groups instead and promote the extension that saves them.
  if((float)P.boostFactor > 1.0f) {
    const auto boostAt = [&](Loc loc) {
      const int p = NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen);
      if(p < 0 || p >= policySize || !isLegal[p])
        return;
      const float boostLogit = maxLogit + std::log((float)P.boostFactor);
      if(boostLogit > policy[p])
        policy[p] = boostLogit;
    };
    //Rescue boost: same long-ladder depth floor as the root gate and the value blend.
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        Loc move = Board::NULL_LOC;
        if(LzLadder::isRealLadderEscape(board, history, loc, nextPlayer, P.minStones, P.minDepth, &move, NULL))
          boostAt(move);
      }
    }
  }

  vector<pair<float,int>> candidates;
  candidates.reserve(policySize);
  for(int pos = 0; pos<policySize; pos++) {
    if(!isLegal[pos])
      continue;
    if(NNPos::isPassPos(pos,nnXLen,nnYLen))
      continue;
    candidates.push_back(make_pair(policy[pos],pos));
  }
  if(candidates.empty())
    return;

  float refLogit = candidates[std::min((size_t)P.penaltyRank, candidates.size()-1)].first;
  const float demoteLogit = refLogit + std::log((float)P.policyFactor);

  int numCandidates = std::min((int)candidates.size(), P.checkNodes);
  std::partial_sort(
    candidates.begin(), candidates.begin() + numCandidates, candidates.end(),
    [](const pair<float,int>& a, const pair<float,int>& b) { return a.first > b.first; }
  );

  for(int i = 0; i<numCandidates; i++) {
    const float logit = candidates[i].first;
    if(logit < maxLogit + std::log((float)P.minPolicy))
      break;
    const int pos = candidates[i].second;
    Loc moveLoc = NNPos::posToLoc(pos,board.x_size,board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC || moveLoc == Board::PASS_LOC)
      continue;

    const int x = Location::getX(moveLoc,board.x_size);
    const int y = Location::getY(moveLoc,board.x_size);
    //Cheap precheck: only moves that touch a group with at most two liberties can change a
    //ladder verdict, so skip the copy and the solver for everything else.
    bool nearAtari = false;
    for(int d = 0; d<4; d++) {
      const int nx = x + NDX[d];
      const int ny = y + NDY[d];
      if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= board.y_size)
        continue;
      Loc nb = Location::getLoc(nx,ny,board.x_size);
      //<=3, not <=2: our move can drop the group by one liberty (or more if it also captures),
      //so a group that ends up in atari may well have had three liberties before the move.
      if(board.colors[nb] != C_EMPTY && board.getNumLiberties(nb) <= 3) {
        nearAtari = true;
        break;
      }
    }
    if(!nearAtari)
      continue;

    Board copy(board);
    BoardHistory copyHist(history);
    if(!copyHist.makeBoardMoveTolerant(copy,moveLoc,nextPlayer))
      continue;
    //Kos make the ladder verdict unreliable (the recapture might be illegal), so skip them.
    if(copy.ko_loc != Board::NULL_LOC)
      continue;

    const int captured = nextPlayer == P_WHITE
      ? (int)(copy.numWhiteCaptures - board.numWhiteCaptures)
      : (int)(copy.numBlackCaptures - board.numBlackCaptures);

    Loc checkedHeads[4];
    int numChecked = 0;
    bool doBoost = false;
    bool doZero = false;
    bool doDemote = false;
    for(int d = 0; d<4; d++) {
      const int nx = x + NDX[d];
      const int ny = y + NDY[d];
      if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= board.y_size)
        continue;
      Loc nb = Location::getLoc(nx,ny,board.x_size);
      Color col = copy.colors[nb];
      if(col == C_EMPTY)
        continue;
      Loc head = copy.chain_head[nb];
      bool seen = false;
      for(int j = 0; j<numChecked; j++) {
        if(checkedHeads[j] == head) { seen = true; break; }
      }
      if(seen)
        continue;
      if(numChecked < 4)
        checkedHeads[numChecked++] = head;

      const int postLibs = copy.getNumLiberties(nb);
      if(postLibs > 2)
        continue;

      if(col != nextPlayer) {
        //Play the chase out from the defender's turn; solver depth d < 0 = captured in -d plies,
        //d > 0 = escaped, d = 0 = unknown.
        int d = 0;
        bool capturedLine = false;
        if(captured == 0 && postLibs <= 1)
          capturedLine = copy.searchIsLadderCaptured(nb,true,buf,&d);
        //Boost: opponent group put in atari and cannot escape; long ladders only.
        //Any group size counts: a single stone in a long ladder is invisible to the value head.
        if(captured == 0 && postLibs == 1 && capturedLine && d <= -P.minDepth)
          doBoost = true;
        //Chase length = remaining escape depth + stones already played; zero when futile.
        if(P.offenseEnabled && captured == 0 && postLibs == 1 && !capturedLine && d > 0) {
          const int ladderCounter = getLadderCounter(copy, nb);
          if(ladderCounter >= P.chaseStones && d + ladderCounter >= P.offenseDepth)
            doZero = true;
        }
      }
      else {
        //Demote: own two-liberty group is captured, line is deep, and costs more than we gain.
        const int chainSize = copy.getChainSize(nb);
        if(postLibs == 2 && chainSize > captured && chainSize >= P.minStones) {
          //Attacker to move on our two-liberty group: same solver, depth included.
          int d = 0;
          if(copy.searchIsLadderCaptured(nb,false,buf,&d)) {
            const int plies = -d;
            if(plies >= P.defenseDepth && (plies/2 + chainSize) > captured)
              doDemote = true;
          }
        }
      }
    }

    if(doBoost && (float)P.boostFactor > 1.0f) {
      const float boostLogit = maxLogit + std::log((float)P.boostFactor);
      if(boostLogit > policy[pos]) {
        //Winrate nudge scaled by the move's own probability.
        if(winMulOut != NULL && P.boostWinrate > 0.0f && sumExp > 0.0f) {
          const float p = std::exp(policy[pos] - maxLogit) / sumExp;
          *winMulOut *= (1.0f + p * (float)P.boostWinrate);
        }
        policy[pos] = boostLogit;
      }
    }
    else if(doZero) {
      //Futile chase: reduce the winrate by the prior, then zero the move.
      if(winMulOut != NULL && P.penaltyWinrate > 0.0f && sumExp > 0.0f) {
        const float p = std::exp(policy[pos] - maxLogit) / sumExp;
        *winMulOut *= (1.0f - p * (float)P.penaltyWinrate);
      }
      policy[pos] = -1e30f;
    }
    else if(doDemote) {
      //Demotion: drop below the penaltyRank-th candidate; penaltyRank == 0 zeroes the move.
      if(P.penaltyRank > 0) {
        if(demoteLogit < policy[pos])
          policy[pos] = demoteLogit;
        if(winMulOut != NULL && P.penaltyWinrate > 0.0f && sumExp > 0.0f) {
          const float p = std::exp(policy[pos] - maxLogit) / sumExp;
          *winMulOut *= (1.0f - p * (float)P.penaltyWinrate);
        }
      }
      else {
        if(winMulOut != NULL && P.penaltyWinrate > 0.0f && sumExp > 0.0f) {
          const float p = std::exp(policy[pos] - maxLogit) / sumExp;
          *winMulOut *= (1.0f - p * (float)P.penaltyWinrate);
        }
        policy[pos] = -1e30f;
      }
    }
  }
}
