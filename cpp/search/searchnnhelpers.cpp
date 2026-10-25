#include "../search/search.h"

#include "../search/searchnode.h"

#include <algorithm>
#include <cmath>
#include <sstream>

//------------------------
#include "../core/using.h"
//------------------------

void Search::computeRootNNEvaluation(NNResultBuf& nnResultBuf, bool includeOwnerMap) {
  Board board = rootBoard;
  const BoardHistory& hist = rootHistory;
  Player pla = rootPla;
  bool skipCache = false;
  bool isRoot = true;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.conservativePassAndIsRoot = searchParams.conservativePass && isRoot;
  nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == pla;
  nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  if(searchParams.ignorePreRootHistory || searchParams.ignoreAllHistory)
    nnInputParams.maxHistory = 0;
  nnInputParams.useDynamicKomi = searchParams.useDynamicKomi || dynamicKomiSet;
  nnInputParams.dynamicKomi = currentDynamicKomi;
  nnEvaluator->evaluateWithLzLadder(
    board, hist, pla, &searchParams.humanSLProfile,
    nnInputParams,
    nnResultBuf, skipCache, includeOwnerMap
  );
}

//LZ-style dynamic komi: choose a virtual komi (fed into the LZ color planes) that keeps white's
//winrate within [dynamicKomiMinWr, dynamicKomiMaxWr], targeting dynamicKomiMidWr. This mirrors
//the handicap/endgame "auto adjust komi" logic from the Leela Zero fork: only the nn input is
//changed, the real rules komi is untouched. Returns true iff the stored virtual komi changed.
bool Search::updateDynamicKomi() {
  //Only Leela Zero-style models carry the color planes this relies on.
  if(nnEvaluator == NULL || !nnEvaluator->isLZModel())
    return false;

  const float targetKomi = searchParams.dynamicKomiTargetKomi;
  const float midWr = searchParams.dynamicKomiMidWr;
  const float minWr = searchParams.dynamicKomiMinWr;
  const float maxWr = searchParams.dynamicKomiMaxWr;

  //Evaluate white's winrate at a trial virtual komi. skipCache=true so tuning probes never
  //populate the nn cache; symmetry 0 keeps the tuning deterministic.
  auto whiteWinrateAtKomi = [&](float komi) -> float {
    MiscNNInputParams nnInputParams;
    nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
    nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
    nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
    nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == rootPla;
    nnInputParams.policyOptimism = searchParams.rootPolicyOptimism;
    if(searchParams.playoutDoublingAdvantage != 0) {
      Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
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

  //White winrate increases monotonically with komi, so bisect between a komi that is winning
  //for white (high) and one that is losing (low), returning the losing side as LZ does.
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

  //KataGo's playoutDoublingAdvantage is inert on V8 inputs, so reuse it as a signed aggression
  //knob and map it linearly into the virtual komi carried by the color planes. Positive means the
  //root player is assumed stronger and should see less komi (and hence play more actively).
  if(searchParams.playoutDoublingAdvantage != 0) {
    const double pda = searchParams.playoutDoublingAdvantage;
    const Player pdaPla = getPlayoutDoublingAdvantagePla();
    const double pdaForRoot = (getOpp(rootPla) == pdaPla) ? -pda : pda;
    const double pdaForWhite = (rootPla == P_WHITE) ? pdaForRoot : -pdaForRoot;
    float newKomi = (float)(rootHistory.rules.komi - 2.5 * pdaForWhite);
    if(newKomi < 0.0f) newKomi = 0.0f;
    if(newKomi > 15.0f) newKomi = 15.0f;

    //Rescue floor. Native KataGo happily plays at extremely low reported winrates (its own
    //auto-PDA handicap games report ~0.05%), so this is not a general "don't be desperate"
    //floor. It only guards the point where the V8 net actually degenerates: in a 4-stone
    //handicap game it starts passing around 0.5% white winrate, while ~0.8% and above still
    //plays normally. Raise K just enough to clear the floor (and stay within [0,15], the domain
    //the color planes were built for), keeping the PDA effect intact wherever possible.
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

MiscNNInputParams Search::paramsForHumanEvaluator(const MiscNNInputParams& nnInputParams) const {
  MiscNNInputParams humanNNInputParams = nnInputParams;
  //Human net always sees the real komi, never the dynamic one.
  humanNNInputParams.useDynamicKomi = false;
  humanNNInputParams.dynamicKomi = 0.0f;
  humanNNInputParams.passAliveSuicideRulesOverride =
    resolveAlwaysComputePassAliveUnderSuicideRules(searchParams, humanEvaluator) ? 1 : 0;
  humanNNInputParams.excludeTerritoryAdjAtariOverride =
    resolveExcludeTerritoryAdjacentToAtari(searchParams, humanEvaluator) ? 1 : 0;
  return humanNNInputParams;
}

bool Search::needsHumanOutputAtRoot() const {
  return humanEvaluator != NULL && (searchParams.humanSLProfile.initialized || !humanEvaluator->requiresSGFMetadata());
}
bool Search::needsHumanOutputInTree() const {
  return needsHumanOutputAtRoot() && (
    searchParams.humanSLPlaExploreProbWeightless > 0 ||
    searchParams.humanSLPlaExploreProbWeightful > 0 ||
    searchParams.humanSLOppExploreProbWeightless > 0 ||
    searchParams.humanSLOppExploreProbWeightful > 0
  );
}

//If isReInit is false, among any threads trying to store, the first one wins
//If isReInit is true, we always replace, even for threads that come later.
//Returns true if a nnOutput was set where there was none before.
bool Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  bool isRoot, bool skipCache, bool isReInit
) {
  bool includeOwnerMap = isRoot || alwaysIncludeOwnerMap;
  bool antiMirrorDifficult = false;
  if(searchParams.antiMirror && mirroringPla != C_EMPTY && mirrorAdvantage >= -0.5 &&
     Location::getCenterLoc(thread.board) != Board::NULL_LOC && thread.board.colors[Location::getCenterLoc(thread.board)] == getOpp(rootPla) &&
     isMirroringSinceSearchStart(thread.history,4) // skip recent 4 ply to be a bit tolerant
  ) {
    includeOwnerMap = true;
    antiMirrorDifficult = true;
  }
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searchParams.drawEquivalentWinsForWhite;
  nnInputParams.conservativePassAndIsRoot = searchParams.conservativePass && isRoot;
  nnInputParams.enablePassingHacks = searchParams.enablePassingHacks;
  nnInputParams.nnPolicyTemperature = searchParams.nnPolicyTemperature;
  nnInputParams.avoidMYTDaggerHack = searchParams.avoidMYTDaggerHackPla == thread.pla;
  nnInputParams.policyOptimism = isRoot ? searchParams.rootPolicyOptimism : searchParams.policyOptimism;
  if(searchParams.playoutDoublingAdvantage != 0) {
    Player playoutDoublingAdvantagePla = getPlayoutDoublingAdvantagePla();
    nnInputParams.playoutDoublingAdvantage = (
      getOpp(thread.pla) == playoutDoublingAdvantagePla ? -searchParams.playoutDoublingAdvantage : searchParams.playoutDoublingAdvantage
    );
  }
  if(searchParams.ignoreAllHistory)
    nnInputParams.maxHistory = 0;
  else if(searchParams.ignorePreRootHistory) {
    nnInputParams.maxHistory = isRoot ? 0 : std::max(0, (int)thread.history.moveHistory.size() - (int)rootHistory.moveHistory.size());
  }
  nnInputParams.useDynamicKomi = searchParams.useDynamicKomi || dynamicKomiSet;
  nnInputParams.dynamicKomi = currentDynamicKomi;
  //Auxiliary evaluation uses the real (rules) komi, never the dynamic virtual komi, and a fixed
  //symmetry so score/ownership reporting is deterministic.
  MiscNNInputParams auxInputParams = nnInputParams;
  auxInputParams.useDynamicKomi = false;
  auxInputParams.dynamicKomi = 0.0f;
  auxInputParams.symmetry = 0;

  std::shared_ptr<NNOutput>* result = NULL;
  std::shared_ptr<NNOutput>* humanResult = NULL;
  std::shared_ptr<NNOutput>* auxResult = NULL;
  //Only compute the auxiliary output if we don't already have one for this node. It is used purely
  //for score/ownership reporting and must never feed back into search values.
  const bool needAuxOutput = auxEvaluator != NULL && (isReInit || node.getAuxOutput() == NULL);
  if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
    result = nnEvaluator->averageMultipleSymmetries(
      thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
      nnInputParams,
      thread.nnResultBuf, includeOwnerMap,
      thread.rand, searchParams.rootNumSymmetriesToSample
    );
    if(needsHumanOutputInTree() || (isRoot && needsHumanOutputAtRoot())) {
      humanResult = humanEvaluator->averageMultipleSymmetries(
        thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
        paramsForHumanEvaluator(nnInputParams),
        thread.nnResultBuf, includeOwnerMap,
        thread.rand, searchParams.rootNumSymmetriesToSample
      );
    }
    if(needAuxOutput) {
      auxEvaluator->evaluate(
        thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
        auxInputParams,
        thread.nnResultBuf, skipCache, true
      );
      auxResult = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
    }
  }
  else {
    nnEvaluator->evaluateWithLzLadder(
      thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
      nnInputParams,
      thread.nnResultBuf, skipCache, includeOwnerMap
    );
    result = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
    if(needsHumanOutputInTree() || (isRoot && needsHumanOutputAtRoot())) {
      humanEvaluator->evaluate(
        thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
        paramsForHumanEvaluator(nnInputParams),
        thread.nnResultBuf, skipCache, includeOwnerMap
      );
      humanResult = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
    }
    if(needAuxOutput) {
      auxEvaluator->evaluate(
        thread.board, thread.history, thread.pla, &searchParams.humanSLProfile,
        auxInputParams,
        thread.nnResultBuf, skipCache, true
      );
      auxResult = new std::shared_ptr<NNOutput>(std::move(thread.nnResultBuf.result));
    }
  }

  if(antiMirrorDifficult) {
    // Copy
    std::shared_ptr<NNOutput>* newNNOutputSharedPtr = new std::shared_ptr<NNOutput>(new NNOutput(**result));
    std::shared_ptr<NNOutput>* tmp = result;
    result = newNNOutputSharedPtr;
    delete tmp;
    hackNNOutputForMirror(*result);
  }

  assert((*result)->noisedPolicyProbs == NULL);
  std::shared_ptr<NNOutput>* noisedResult = maybeAddPolicyNoiseAndTemp(thread,isRoot,result->get());
  if(noisedResult != NULL) {
    std::shared_ptr<NNOutput>* tmp = result;
    result = noisedResult;
    delete tmp;
  }

  node.nodeAge.store(searchNodeAge,std::memory_order_release);
  //If this is a re-initialization of the nnOutput, we don't want to add any visits or anything.
  //Also don't bother updating any of the stats. Technically we should do so because winLossValueSum
  //and such will have changed potentially due to a new orientation of the neural net eval
  //slightly affecting the evals, but this is annoying to recompute from scratch, and on the next
  //visit updateStatsAfterPlayout should fix it all up anyways.
  if(isReInit) {
    if(humanResult != NULL)
      node.storeHumanOutput(humanResult,thread); // ignore the wasNullBefore from this one
    if(auxResult != NULL)
      node.storeAuxOutput(auxResult,thread); // ignore the wasNullBefore from this one
    bool wasNullBefore = node.storeNNOutput(result,thread);
    return wasNullBefore;
  }
  else {
    // Store human result first, so that the presence of the main result guarantees
    // that the human result exists in the case we have a human evaluator.
    if(auxResult != NULL) {
      bool auxSuc = node.storeAuxOutputIfNull(auxResult);
      if(!auxSuc)
        delete auxResult;
    }
    if(humanResult != NULL) {
      bool humanSuc = node.storeHumanOutputIfNull(humanResult);
      if(!humanSuc)
        delete humanResult;
    }
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc)
      delete result;
    else {
      addCurrentNNOutputAsLeafValue(node,true);
    }
    return suc;
  }
}


//Assumes node already has an nnOutput
bool Search::maybeRecomputeExistingNNOutput(
  SearchThread& thread, SearchNode& node, bool isRoot
) {
  bool recomputeHappened = false;
  //Right now only the root node currently ever needs to recompute, and only if it's old
  if(isRoot && node.nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    //See if we're the lucky thread that gets to do the update!
    //Threads that pass by here later will NOT wait for us to be done before proceeding with search.
    //We accept this and tolerate that for a few iterations potentially we will be using the OLD policy - without noise,
    //or without root temperature, etc.
    //Or if we have none of those things, then we'll not end up updating anything except the age, which is okay too.
    uint32_t oldAge = node.nodeAge.exchange(searchNodeAge,std::memory_order_acq_rel);
    if(oldAge < searchNodeAge) {
      NNOutput* nnOutput = node.getNNOutput();
      NNOutput* humanOutput = node.getHumanOutput();
      testAssert(nnOutput != NULL);

      //Recompute if we have no ownership map, since we need it for getEndingWhiteScoreBonus
      //If conservative passing, then we may also need to recompute the root policy ignoring the history if a pass ends the game
      //If averaging a bunch of symmetries, then we need to recompute it too
      //If root needs different optimism, we need to recompute it.
      //If humanSL is missing, but we want it, we need to recompute.
      //Also do so when ignoring history pre root
      if(nnOutput->whiteOwnerMap == NULL ||
         (searchParams.conservativePass && thread.history.passWouldEndGame(thread.board,thread.pla)) ||
         searchParams.rootNumSymmetriesToSample > 1 ||
         searchParams.rootPolicyOptimism != searchParams.policyOptimism ||
         (searchParams.ignorePreRootHistory && !searchParams.ignoreAllHistory) ||
         (humanOutput == NULL && needsHumanOutputAtRoot())
      ) {
        //We *can* use cached evaluations even though parameters are changing, because:
        //conservativePass is part of the nn hash
        //Symmetry averaging skips the cache on its own when it does symmetry sampling without replacement
        //The optimism is part of the nn hash
        //When pre-root history is ignored at the root, maxHistory is 0 and the nn cache distinguishes 0 from nonzero.
        const bool skipCache = false;
        initNodeNNOutput(thread,node,isRoot,skipCache,true);
        recomputeHappened = true;
      }
      //We also need to recompute the root nn if we have root noise or temperature and that's missing.
      else {
        //We don't need to go all the way to the nnEvaluator, we just need to maybe add those transforms
        //to the existing policy.
        std::shared_ptr<NNOutput>* result = maybeAddPolicyNoiseAndTemp(thread,isRoot,nnOutput);
        if(result != NULL) {
          node.storeNNOutput(result,thread);
          recomputeHappened = true;
        }
      }
    }
  }
  return recomputeHappened;
}
