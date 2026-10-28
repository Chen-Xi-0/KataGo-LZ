//LZ fork auxiliary (score/ownership) evaluator helpers, split out of search/searchnnhelpers.cpp
//(see 05_验证输出/LZ分支代码解耦重构设计_20261028.md §4.3). The aux evaluator is experimental and
//fork-only; it never feeds policy into the search. The SearchNode/SearchNodeStats aux fields and
//the stats plumbing stay in their upstream files (struct fields cannot be externalized); this
//module concentrates the pure logic that was fork-added: input-parameter shaping and the
//experimental auxValueWeight value blend.

#ifndef KATAGO_LZAUX_H
#define KATAGO_LZAUX_H

#include "../core/global.h"
#include "../neuralnet/nneval.h"

namespace LzAux {

//Input parameters for the aux evaluator: it always sees the real komi (no dynamic komi folded
//into the color planes) and a fixed symmetry, so its score/ownership are position-canonical.
MiscNNInputParams makeAuxInputParams(const MiscNNInputParams& baseParams);

//Experimental aux value mixing: blend the auxiliary model's value into the main model's value.
//The policy (and everything policy-driven) stays the main model's, so move selection remains
//unchanged; only winrate/score/lead are mixed. auxValueWeight = 0 disables this. No-op when
//auxResult is NULL. Takes ownership of *result (replaces it with the blended copy and deletes
//the old shared_ptr).
void blendAuxValue(
  std::shared_ptr<NNOutput>*& result,
  std::shared_ptr<NNOutput>* auxResult,
  double auxValueWeight
);

}

#endif //KATAGO_LZAUX_H
