//LZ ladder correction: V8 inputs have no ladder planes, so policy and value are corrected
//from solver verdicts (policy surgery) and from evaluating the resolved capture (value blend).
//LZ models only.

#ifndef KATAGO_LZLADDER_H
#define KATAGO_LZLADDER_H

#include <atomic>
#include <string>

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

//Loaded from cfg by LzSetup::loadLzLadderParams.
namespace LzLadder {
  struct Params {
    bool enabled;        //lzLadderEnabled: master switch
    int minStones;       //lzLadderMinStones: minimum chased-group size
    int checkNodes;      //lzLadderCheckNodes: policy candidates checked for demotion
    double minPolicy;    //lzLadderMinPolicy: candidate policy floor (probability space)
    double policyFactor; //lzLadderPolicyFactor: demote-to factor below the penaltyRank move
    int penaltyRank;     //lzLadderPenaltyBase: rank index the demoted move drops below
    int chaseStones;     //lzLadderChaseStones: min ladder stones already played for chase zeroing
    double boostFactor;  //lzLadderBoostFactor: >1 lifts a winning ladder capture above the top
    int defenseDepth;    //lzLadderDefenseDepth: min plies of the losing line before demotion
    bool offenseEnabled; //lzLadderOffenseEnabled: enable the futile-chase zeroing
    int offenseDepth;    //lzLadderOffenseDepth: chase length that counts as futile
    double boostWinrate; //lzLadderBoostWinrate: winrate nudge factor on a boosted capture
    double penaltyWinrate;//lzLadderPenaltyWinrate: winrate nudge factor on a demoted/zeroed move
    int minDepth;        //lzLadderMinDepth: min remaining escape depth (plies) for the value correction
    double ladderValueBlend;//lzLadderEscapedValueBlend: blend for a solver-certified escaped group

    Params();
  };
  extern Params params;
  //Set while the root position has a ladder-relevant shape.
  extern std::atomic<int> rootArmed;

  //Real ladder escape: own one-liberty group, only liberty extends to exactly two liberties
  //without ko, solver says it cannot be captured, and the escape is at least minDepth deep.
  bool isRealLadderEscape(
    const Board& board, const BoardHistory& history, Loc loc, Player pla,
    int minStones, int minDepth, Loc* outMove, int* outDepth
  );

  //Same certification for a group that already reached two liberties.
  bool isEscapedTwoLibertyGroup(
    const Board& board, Loc loc, Player pla, int minStones, int minDepth,
    Loc* outMove, int* outDepth
  );

  //Sets armedOut to 1 if `pla` has an escapable long ladder at the root.
  void armRootScan(
    const Board& board, const BoardHistory& hist, Player pla, std::atomic<int>& armedOut
  );
}

//Policy surgery from solver verdicts; *winMulOut carries the winrate nudge (1.0 = none).
void applyLZLadderCorrectionV5(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  int nnXLen,
  int nnYLen,
  float* policy,
  const bool* isLegal,
  float* winMulOut
);

#endif //KATAGO_LZLADDER_H
