//LZ fork ladder correction (v5), split out of neuralnet/nneval.cpp so upstream keeps only
//two hook lines there (see 05_验证输出/LZ分支代码解耦重构设计_20261028.md §4.2).
//Leela Zero-style inputs (V8) have no ladder feature planes, so both the policy and the value
//head are blind to ladders. Instead of hand-tuned penalties:
//  - KataGo's rule-based solver provides exact verdicts (which two-liberty groups are
//    capturable / escapable) and the chase moves;
//  - for a proven-capturable big group the ladder line is played out on the board and the
//    RESOLVED position is evaluated with the SAME net - after the capture the stones are
//    simply gone, so even a ladder-blind net reads the material correctly. That winrate,
//    blended back in, replaces every hand-tuned penalty magnitude (design doc: 05_验证输出/
//    征子修正v3_对齐参考实现设计_20261025.md, v5 section);
//  - policy surgery keeps the reference fork's mild semantics: demote moves that leave a
//    solver-dead group (just below the 3rd-ranked move, probability factor 0.9) and zero
//    futile chases of solver-alive groups, and a boost for escapes the solver proves alive.
//Only used for LZ models; native KataGo nets are unaffected.

#ifndef KATAGO_LZLADDER_H
#define KATAGO_LZLADDER_H

#include <atomic>
#include <string>

#include "../core/config_parser.h"
#include "../core/global.h"
#include "../game/board.h"
#include "../game/boardhistory.h"

//Parameters of the LZ ladder correction (solver verdict + policy surgery + escaped-group value
//blend). Loaded from cfg by LzSetup::loadLzLadderParams; only takes effect for LZ models
//(isLZModelFlag).
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
  //Set by the search when the ROOT position contains a ladder-relevant shape (a group with one or
  //two liberties whose solver verdict matters). The escaped-group value correction only runs while
  //this is set, so ordinary searches pay nothing for it.
  extern std::atomic<int> rootArmed;

  //Shared certification of "a real ladder escape", used by the rescue boost, the root arming scan
  //and the escaped-group value blend so all three apply the same predicate:
  //  - the group is `pla`'s, has exactly one liberty, and at least `minStones` stones;
  //  - its extension is the ONLY empty neighbor, reaches exactly two liberties and creates no ko;
  //  - the solver, attacker to move, says it cannot be captured and the verdict is known
  //    (depth > 0; the solver's node budget is upstream's fixed 25000, exhausted => 0);
  //  - that escape is a long ladder, i.e. the remaining depth is at least `minDepth` plies.
  //*outMove receives the extension, *outDepth the solver's escape depth.
  bool isRealLadderEscape(
    const Board& board, const BoardHistory& history, Loc loc, Player pla,
    int minStones, int minDepth, Loc* outMove, int* outDepth
  );

  //Same certification for a group that already reached two liberties ("the escape already
  //happened"). *outMove receives one of its liberties.
  bool isEscapedTwoLibertyGroup(
    const Board& board, Loc loc, Player pla, int minStones, int minDepth,
    Loc* outMove, int* outDepth
  );

  //Shared certification used by the search's root arming scan (searchnnhelpers.cpp) and the
  //kata-ladder diagnostic (gtp.cpp): for every 1- or 2-liberty group of `pla`, run the bounded
  //solver and report capture verdict plus the attacker moves that work (for libs == 2).
  void armRootScan(
    const Board& board, const BoardHistory& hist, Player pla, std::atomic<int>& armedOut
  );
}

//Policy half of the ladder correction: solver-verdict demotions on the scaled logits (before the
//softmax, legality mask in place), plus the boost for a proven ladder capture. Writes the
//multiplicative winrate nudge to *winMulOut (1.0 = none). Called from NNEvaluator::evaluate's
//postprocessing, right before the policy softmax.
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
