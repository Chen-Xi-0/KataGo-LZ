//LZ fork setup helpers, split out of program/setup.cpp and command/gtp.cpp so the
//upstream files keep only one-line call sites (see 05_验证输出/LZ分支代码解耦重构设计_20261028.md).
//Everything here is specific to Leela Zero text weight files; native KataGo models never
//reach any of this code.

#ifndef KATAGO_LZSETUP_H
#define KATAGO_LZSETUP_H

#include "../core/config_parser.h"
#include "../search/searchparams.h"

namespace LzSetup {

//Sniff whether the model file is a Leela Zero text weight file. Light-weight: reads and
//(maybe) gunzips the file once, no model construction. Used to auto-configure LZ-friendly
//search defaults: the LZ weights are not exactly equivariant under the 8 symmetries, so
//per-node random symmetries add noise that the search amplifies into chaotic move
//selection (see 05_验证输出/对弈全败根因_nnRandomize噪声_20261025.md), and the
//chosen-move temperature randomizes away the best move in genmove (analysis mode ignores
//it, so analysis-based testing does not see it).
bool isLZWeightFile(const std::string& fileName);

//LZ ladder correction v5 parameters (see nneval.h). Only effective for LZ models.
//Call once at startup, before any search runs; LzLadder::params is read-only afterwards.
void loadLzLadderParams(ConfigParser& cfg);

//LZ dynamic komi knobs (useDynamicKomi, dynamicKomiMaxWr/MinWr/MidWr/TargetKomi). Same
//startup call as loadLzLadderParams; the controller itself lives in search/lzdynkomi.cpp.
void loadDynamicKomiParams(ConfigParser& cfg, SearchParams& params);

//LZ models: zero the genmove move-selection temperature (defaults 0.1, early 0.5).
//Analysis mode ignores these by design, so analysis-based strength testing does not
//see them; genmove play without this gets randomized away from the best move. cfg
//explicit values still win. nnRandomize is handled in Setup::initializeNNEvaluators.
//NOTE: deliberately NOT gated on cfg's nnRandomize - passing nnRandomize=false (which
//lizzieyzy and every match script does) used to disable this block as a side effect, leaving
//genmove sampling at the 0.1/0.5 temperature defaults while analysis mode ignored it. That is
//exactly the "genmove looks weaker than analysis at equal compute" trap.
void applyLzGenmoveTemperatureDefaults(
  const ConfigParser& config,
  const std::string& nnModelFile,
  double& chosenMoveTemperature,
  double& chosenMoveTemperatureEarly
);

}

#endif //KATAGO_LZSETUP_H
