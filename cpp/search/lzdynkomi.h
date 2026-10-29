//Dynamic komi controller: chooses the virtual komi folded into the LZ color planes.

#ifndef KATAGO_LZDYNKOMI_H
#define KATAGO_LZDYNKOMI_H

#include "../core/logger.h"
#include "../neuralnet/nneval.h"
#include "../search/searchparams.h"

namespace LzDynKomi {

//Updates the virtual komi for the root position. Returns true if it changed (caller clears
//the tree). No-op for non-LZ models.
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
