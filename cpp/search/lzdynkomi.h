//LZ fork dynamic komi (virtual-komi controller), split out of search/searchnnhelpers.cpp so
//the upstream file keeps only a one-line member delegate (see 05_验证输出/LZ分支代码解耦重构设计_20261028.md §4.2b).
//The member function Search::updateDynamicKomi forwards here; the state (currentDynamicKomi,
//dynamicKomiSet) stays on Search and is passed in/out explicitly, so this module never includes
//search.h and cannot create a dependency cycle.

#ifndef KATAGO_LZDYNKOMI_H
#define KATAGO_LZDYNKOMI_H

#include "../core/logger.h"
#include "../neuralnet/nneval.h"
#include "../search/searchparams.h"

namespace LzDynKomi {

//Recompute the virtual komi folded into the LZ-compatible net inputs (the V8 color planes).
//Parameters are exactly the Search state the controller reads/writes:
//  nnEvaluator        - the main net (must be an LZ model; otherwise this is a no-op)
//  searchParams       - the current search parameters (PDA + dynamicKomi* knobs)
//  rootBoard/rootHistory/rootPla - the root position being evaluated
//  pdaPla             - Search::getPlayoutDoublingAdvantagePla()
//  logger             - for the "Dynamic komi ..." messages
//  currentDynamicKomi/dynamicKomiSet - in/out state; set currentDynamicKomi when it changes.
//Returns true iff the stored virtual komi changed (the caller then drops the search tree).
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
);

}

#endif //KATAGO_LZDYNKOMI_H
