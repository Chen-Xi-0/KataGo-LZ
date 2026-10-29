//Aux score/ownership evaluator helpers: input shaping and the auxValueWeight value blend.

#ifndef KATAGO_LZAUX_H
#define KATAGO_LZAUX_H

#include "../core/global.h"
#include "../neuralnet/nneval.h"

namespace LzAux {

//Aux input params: real komi (no dynamic komi) and symmetry 0.
MiscNNInputParams makeAuxInputParams(const MiscNNInputParams& baseParams);

//Blend the auxiliary model's winrate/score/lead into the main value; policy is unchanged.
//No-op when auxResult is NULL or weight is 0.
void blendAuxValue(
  std::shared_ptr<NNOutput>*& result,
  std::shared_ptr<NNOutput>* auxResult,
  double auxValueWeight
);

}

#endif //KATAGO_LZAUX_H
