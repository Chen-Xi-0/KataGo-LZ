//Leela Zero weight setup helpers.

#ifndef KATAGO_LZSETUP_H
#define KATAGO_LZSETUP_H

#include "../core/config_parser.h"
#include "../search/searchparams.h"

namespace LzSetup {

//True if the file is an LZ text weight file (read + maybe gunzip + format check).
bool isLZWeightFile(const std::string& fileName);

//Load lzLadder* keys; call once at startup.
void loadLzLadderParams(ConfigParser& cfg);

//Load useDynamicKomi and dynamicKomi* keys.
void loadDynamicKomiParams(ConfigParser& cfg, SearchParams& params);

//Zero genmove temperature for LZ weights unless the config sets it explicitly.
void applyLzGenmoveTemperatureDefaults(
  const ConfigParser& config,
  const std::string& nnModelFile,
  double& chosenMoveTemperature,
  double& chosenMoveTemperatureEarly
);

}

#endif //KATAGO_LZSETUP_H
