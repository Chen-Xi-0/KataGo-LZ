#include "../search/lzaux.h"

#include <algorithm>

using namespace std;

namespace LzAux {

MiscNNInputParams makeAuxInputParams(const MiscNNInputParams& baseParams) {
  MiscNNInputParams auxInputParams = baseParams;
  auxInputParams.useDynamicKomi = false;
  auxInputParams.dynamicKomi = 0.0f;
  auxInputParams.symmetry = 0;
  return auxInputParams;
}

void blendAuxValue(
  std::shared_ptr<NNOutput>*& result,
  std::shared_ptr<NNOutput>* auxResult,
  double auxValueWeight
) {
  if(auxResult != NULL && *auxResult != NULL && auxValueWeight > 0.0) {
    const double w = std::min(1.0, std::max(0.0, auxValueWeight));
    std::shared_ptr<NNOutput> blended = std::make_shared<NNOutput>(**result);
    const NNOutput& a = **auxResult;
    blended->whiteWinProb = (float)((1.0-w)*blended->whiteWinProb + w*a.whiteWinProb);
    blended->whiteLossProb = (float)((1.0-w)*blended->whiteLossProb + w*a.whiteLossProb);
    blended->whiteNoResultProb = (float)((1.0-w)*blended->whiteNoResultProb + w*a.whiteNoResultProb);
    blended->whiteScoreMean = (float)((1.0-w)*blended->whiteScoreMean + w*a.whiteScoreMean);
    blended->whiteScoreMeanSq = (float)((1.0-w)*blended->whiteScoreMeanSq + w*a.whiteScoreMeanSq);
    blended->whiteLead = (float)((1.0-w)*blended->whiteLead + w*a.whiteLead);
    std::shared_ptr<NNOutput>* tmp = result;
    result = new std::shared_ptr<NNOutput>(blended);
    delete tmp;
  }
}

}
