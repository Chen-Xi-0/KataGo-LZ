#include "../search/lzdynkomi.h"

#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../core/global.h"

#include <cmath>
#include <sstream>
#include <vector>

using namespace std;

namespace LzDynKomi {

bool updateDynamicKomi(
  NNEvaluator* nnEvaluator,
  const SearchParams& searchParams,
  const Board& rootBoard,
  const BoardHistory& rootHistory,
  Player rootPla,
  Player pdaPla,
  Logger* logger,
  float& currentDynamicKomi,
  bool& dynamicKomiSet
) {
  //Only Leela Zero-style models carry the color planes this relies on.
  if(nnEvaluator == NULL || !nnEvaluator->isLZModel())
    return false;

  const float targetKomi = searchParams.dynamicKomiTargetKomi;
  const float midWr = searchParams.dynamicKomiMidWr;
  const float minWr = searchParams.dynamicKomiMinWr;
  const float maxWr = searchParams.dynamicKomiMaxWr;

  //Trial eval: skip the cache and use symmetry 0.
  auto whiteWinrateAtKomi = [&](float komi) -> float {
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
    nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
    nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
    nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == rootPla;
    nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
    if(searchParams.playoutDoublingAdvantage != 0) {
      Player playoutDoublingAdvantagePla = pdaPla;
      nnInputParams.playoutDoublingAdvantage = (
        getOpp(rootPla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
      );
    }
    if(searchParams.ignorePreRootHistory || searchParams.ignoreAllHistory)
      nnInputParams.maxHistory = 0;
    nnInputParams.useDynamicKomi = true;
    nnInputParams.dynamicKomi = komi;
    nnInputParams.symmetry = 0;

    NNResultBuf buf;
    nnEvaluator->evaluate(
      rootBoard, rootHistory, rootPla, &searchParams.humanSLProfile,
      nnInputParams, buf, true, false
    );
    return buf.result->whiteWinProb;
  };

  //Bisect between a winning and a losing komi; return the losing side.
  auto binarySearchKomi = [&](float high, float low, float target) -> float {
    for(int i = 0; i<8; i++) {
      float k = (high + low) / 2.0f;
      if(whiteWinrateAtKomi(k) > target)
        high = k;
      else
        low = k;
    }
    return low;
  };

  //PDA is inert on V8 inputs; map it linearly to a virtual komi (positive = stronger).
  if(searchParams.playoutDoublingAdvantage != 0) {
    const double pda = searchParams.playoutDoublingAdvantage;
    const Player pdaPlaArg = pdaPla;
    const double pdaForRoot = (getOpp(rootPla) == pdaPlaArg) ? -pda : pda;
    const double pdaForWhite = (rootPla == P_WHITE) ? pdaForRoot : -pdaForRoot;
    float newKomi = (float)(rootHistory.rules.komi - 2.5 * pdaForWhite);
    if(newKomi < 0.0f) newKomi = 0.0f;
    if(newKomi > 15.0f) newKomi = 15.0f;

    //Rescue floor: raise K just enough that the net does not degenerate (~0.5% winrate).
    const float rescueFloor = 0.02f;
    if(newKomi < rootHistory.rules.komi && whiteWinrateAtKomi(newKomi) < rescueFloor) {
      const float maxKomi = 15.0f;
      float komi = newKomi;
      bool ok = false;
      for(int i = 0; i<8 && komi < maxKomi; i++) {
        float next = 2.0f*komi + 1.0f;
        if(!(next > komi))
          next = komi + 1.0f;
        if(next > maxKomi)
          next = maxKomi;
        komi = next;
        if(whiteWinrateAtKomi(komi) >= rescueFloor) { ok = true; break; }
      }
      newKomi = ok ? binarySearchKomi(komi, newKomi, rescueFloor) : komi;
    }

    if(dynamicKomiSet && newKomi == currentDynamicKomi)
      return false;
    currentDynamicKomi = newKomi;
    dynamicKomiSet = true;
    {
      std::ostringstream msg;
      msg << "Dynamic komi set to " << newKomi << " from playoutDoublingAdvantage " << pda;
      logger->write(msg.str());
    }
    return true;
  }

  if(!searchParams.useDynamicKomi) {
    if(!dynamicKomiSet)
      return false;
    //Turning dynamic komi off also changes the stored virtual komi, so drop the old tree.
    dynamicKomiSet = false;
    return true;
  }
  if(maxWr <= minWr)
    return false;
  if(minWr < 0.0f || maxWr > 1.0f)
    return false;

  const float origKomi = rootHistory.rules.komi;
  float newKomi = origKomi;
  const float rootEval = whiteWinrateAtKomi(origKomi);
  if(rootEval < minWr) {
    //White is losing badly: raise the virtual komi (doubling) until white is no longer desperate,
    //then bisect back towards the neutral point.
    float komi = origKomi;
    bool ok = false;
    for(int i = 0; i<32; i++) {
      float next = 2.0f*komi - (targetKomi - 7.5f);
      //Guard against a non-increasing sequence (e.g. komi == 0 with a 7.5 neutral point).
      if(!(next > komi))
        next = komi + std::max(1.0f, std::fabs(komi));
      komi = next;
      if(whiteWinrateAtKomi(komi) >= midWr) { ok = true; break; }
      if(komi > 500.0f) break;
    }
    if(ok)
      newKomi = binarySearchKomi(komi, (komi + targetKomi - 7.5f) / 2.0f, midWr);
  }
  else if(rootEval > maxWr && origKomi != targetKomi && origKomi > targetKomi) {
    //White is winning too comfortably: lower the virtual komi towards the neutral point if that
    //still leaves white ahead; otherwise bisect to hold white at the target winrate.
    if(whiteWinrateAtKomi(targetKomi) < midWr)
      newKomi = binarySearchKomi(origKomi, targetKomi, midWr);
    else
      newKomi = targetKomi;
  }

  if(dynamicKomiSet && newKomi == currentDynamicKomi)
    return false;
  currentDynamicKomi = newKomi;
  dynamicKomiSet = true;
  {
    std::ostringstream msg;
    msg << "Dynamic komi adjusted to " << newKomi << " (root white winrate at real komi " << rootEval << ")";
    logger->write(msg.str());
  }
  return true;
}

}
