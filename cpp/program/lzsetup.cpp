#include "../program/lzsetup.h"

#include "../core/fileutils.h"
#include "../core/global.h"
#include "../neuralnet/lzconverter.h"
#include "../neuralnet/nneval.h"

using namespace std;

namespace LzSetup {

bool isLZWeightFile(const string& fileName) {
  string lower = Global::toLower(fileName);
  if(Global::isSuffix(lower,".onnx") || Global::isSuffix(lower,".onnx.gz"))
    return false;
  try {
    string uncompressed;
    string sha256Buf;
    if(Global::isSuffix(lower,".txt") || Global::isSuffix(lower,".bin"))
      FileUtils::loadFileIntoString(fileName,"",uncompressed,&sha256Buf);
    else if(Global::isSuffix(lower,".gz"))
      FileUtils::uncompressAndLoadFileIntoString(fileName,"",uncompressed,&sha256Buf);
    else
      return false;
    //Only plain text candidates are LZ (.bin/.bin.gz are binary KataGo floats).
    if(Global::isSuffix(lower,".bin") || Global::isSuffix(lower,".bin.gz"))
      return false;
    return LZConverter::looksLikeLZ(uncompressed);
  }
  catch(const std::exception&) {
    return false;
  }
}

void loadLzLadderParams(ConfigParser& cfg) {
  LzLadder::Params& p = LzLadder::params;
  if(cfg.contains("lzLadderEnabled")) p.enabled = cfg.getBool("lzLadderEnabled");
  if(cfg.contains("lzLadderMinStones")) p.minStones = cfg.getInt("lzLadderMinStones",1,361);
  if(cfg.contains("lzLadderCheckNodes")) p.checkNodes = cfg.getInt("lzLadderCheckNodes",1,362);
  if(cfg.contains("lzLadderMinPolicy")) p.minPolicy = cfg.getDouble("lzLadderMinPolicy",1e-9,1.0);
  if(cfg.contains("lzLadderPolicyFactor")) p.policyFactor = cfg.getDouble("lzLadderPolicyFactor",1e-6,1.0);
  if(cfg.contains("lzLadderPenaltyBase")) p.penaltyRank = cfg.getInt("lzLadderPenaltyBase",0,361);
  if(cfg.contains("lzLadderChaseStones")) p.chaseStones = cfg.getInt("lzLadderChaseStones",1,361);
  if(cfg.contains("lzLadderBoostFactor")) p.boostFactor = cfg.getDouble("lzLadderBoostFactor",1.0,100.0);
  if(cfg.contains("lzLadderDefenseDepth")) p.defenseDepth = cfg.getInt("lzLadderDefenseDepth",1,400);
  if(cfg.contains("lzLadderOffenseEnabled")) p.offenseEnabled = cfg.getBool("lzLadderOffenseEnabled");
  if(cfg.contains("lzLadderOffenseDepth")) p.offenseDepth = cfg.getInt("lzLadderOffenseDepth",1,400);
  if(cfg.contains("lzLadderBoostWinrate")) p.boostWinrate = cfg.getDouble("lzLadderBoostWinrate",0.0,10.0);
  if(cfg.contains("lzLadderPenaltyWinrate")) p.penaltyWinrate = cfg.getDouble("lzLadderPenaltyWinrate",0.0,10.0);
  if(cfg.contains("lzLadderMinDepth")) p.minDepth = cfg.getInt("lzLadderMinDepth",1,400);
  if(cfg.contains("lzLadderEscapedValueBlend")) p.ladderValueBlend = cfg.getDouble("lzLadderEscapedValueBlend",0.0,1.0);
}

void loadDynamicKomiParams(ConfigParser& cfg, SearchParams& params) {
  const string idxStr = "";  //single-model loading: no per-model index suffix
  if(cfg.contains("useDynamicKomi"+idxStr)) params.useDynamicKomi = cfg.getBool("useDynamicKomi"+idxStr);
  else if(cfg.contains("useDynamicKomi"))   params.useDynamicKomi = cfg.getBool("useDynamicKomi");
  else                                      params.useDynamicKomi = false;
  if(cfg.contains("dynamicKomiMaxWr"+idxStr))
    params.dynamicKomiMaxWr = cfg.getFloat("dynamicKomiMaxWr"+idxStr,0.0f,1.0f);
  else if(cfg.contains("dynamicKomiMaxWr"))
    params.dynamicKomiMaxWr = cfg.getFloat("dynamicKomiMaxWr",0.0f,1.0f);
  else
    params.dynamicKomiMaxWr = 0.5f;
  if(cfg.contains("dynamicKomiMinWr"+idxStr))
    params.dynamicKomiMinWr = cfg.getFloat("dynamicKomiMinWr"+idxStr,0.0f,1.0f);
  else if(cfg.contains("dynamicKomiMinWr"))
    params.dynamicKomiMinWr = cfg.getFloat("dynamicKomiMinWr",0.0f,1.0f);
  else
    params.dynamicKomiMinWr = 0.1f;
  if(cfg.contains("dynamicKomiMidWr"+idxStr))
    params.dynamicKomiMidWr = cfg.getFloat("dynamicKomiMidWr"+idxStr,0.0f,1.0f);
  else if(cfg.contains("dynamicKomiMidWr"))
    params.dynamicKomiMidWr = cfg.getFloat("dynamicKomiMidWr",0.0f,1.0f);
  else
    params.dynamicKomiMidWr = 0.4f;
  if(cfg.contains("dynamicKomiTargetKomi"+idxStr))
    params.dynamicKomiTargetKomi = cfg.getFloat("dynamicKomiTargetKomi"+idxStr,-100.0f,100.0f);
  else if(cfg.contains("dynamicKomiTargetKomi"))
    params.dynamicKomiTargetKomi = cfg.getFloat("dynamicKomiTargetKomi",-100.0f,100.0f);
  else
    params.dynamicKomiTargetKomi = 7.5f;
}

void applyLzGenmoveTemperatureDefaults(
  const ConfigParser& config,
  const string& nnModelFile,
  double& chosenMoveTemperature,
  double& chosenMoveTemperatureEarly
) {
  if(!config.contains("chosenMoveTemperature")
     && !config.contains("chosenMoveTemperatureEarly")
     && isLZWeightFile(nnModelFile)) {
    chosenMoveTemperature = 0.0;
    chosenMoveTemperatureEarly = 0.0;
  }
}

}
