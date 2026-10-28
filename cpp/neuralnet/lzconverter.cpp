#include "lzconverter.h"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "../core/global.h"
#include "desc.h"

namespace {

//Standard LZ/ELF constants for plain 19x19 SINGLE-head nets.
const int LZ_INPUT_PLANES = 18;
const int NUM_INTERSECTIONS = 361;
const int BOARD_SIZE = 19;
const int POTENTIAL_MOVES = 362;   //361 + pass
const int POL_FC_IN = 2 * NUM_INTERSECTIONS;
const int VALUE_HIDDEN = 256;
//LZ batch norm epsilon (Network.cpp: process_bn_var), float32 as in the LZ engine.
const float LZ_BN_EPS = 1e-5f;
//KataGo's own BN epsilon used for the folded layers.
const float KG_BN_EPS = 1e-20f;

inline bool isSpace(char c) {
  return c == ' ' || c == '\t' || c == '\r';
}

//Splits the input into non-empty lines, trimming trailing whitespace.
struct LineReader {
  const std::string& data;
  size_t pos;
  int lineNo;

  explicit LineReader(const std::string& d) : data(d), pos(0), lineNo(0) {}

  bool nextLine(const char*& b, const char*& e) {
    while(pos < data.size()) {
      const size_t start = pos;
      const size_t nl = data.find('\n', start);
      const size_t rawEnd = (nl == std::string::npos) ? data.size() : nl;
      pos = (nl == std::string::npos) ? data.size() : nl + 1;
      lineNo++;
      size_t s = start;
      size_t t = rawEnd;
      while(s < t && isSpace(data[s])) s++;
      while(t > s && isSpace(data[t - 1])) t--;
      if(s < t) {
        b = data.data() + s;
        e = data.data() + t;
        return true;
      }
    }
    return false;
  }
};

struct Tokenizer {
  const char* p;
  const char* e;

  Tokenizer(const char* b, const char* end) : p(b), e(end) {}

  bool next(const char*& tb, const char*& te) {
    while(p < e && isSpace(*p)) p++;
    if(p >= e) return false;
    tb = p;
    while(p < e && !isSpace(*p)) p++;
    te = p;
    return true;
  }
};

size_t countTokens(const char* b, const char* e) {
  size_t n = 0;
  Tokenizer tk(b, e);
  const char* tb;
  const char* te;
  while(tk.next(tb, te)) n++;
  return n;
}

float parseFloatToken(const char* tb, const char* te, int lineNo, const char* what) {
  float v = 0.0f;
  const std::from_chars_result res = std::from_chars(tb, te, v, std::chars_format::general);
  if(res.ec != std::errc() || res.ptr != te)
    throw StringError(
      "Leela Zero weights: line " + Global::intToString(lineNo) + ": " + what + " contains a non-numeric token"
    );
  if(!std::isfinite(v))
    throw StringError(
      "Leela Zero weights: line " + Global::intToString(lineNo) + ": " + what + " contains a non-finite value"
    );
  return v;
}

int parseFirstInteger(const char* b, const char* e, int& out) {
  Tokenizer tk(b, e);
  const char* tb;
  const char* te;
  if(!tk.next(tb, te)) return false;
  int v = 0;
  const std::from_chars_result res = std::from_chars(tb, te, v);
  if(res.ec != std::errc() || res.ptr != te) return false;
  out = v;
  return true;
}

struct LineView {
  const char* b;
  const char* e;
  int lineNo;
};

LineView nextLineOrThrow(LineReader& r, const char* what) {
  LineView lv;
  if(!r.nextLine(lv.b, lv.e))
    throw StringError(std::string("Leela Zero weights: unexpected end of file while reading ") + what);
  lv.lineNo = r.lineNo;
  return lv;
}

void parseFloatLine(const LineView& lv, std::vector<float>& out, const char* what) {
  const size_t n = countTokens(lv.b, lv.e);
  out.resize(n);
  Tokenizer tk(lv.b, lv.e);
  const char* tb;
  const char* te;
  size_t i = 0;
  while(tk.next(tb, te)) out[i++] = parseFloatToken(tb, te, lv.lineNo, what);
}

void parseFloatLine(const LineView& lv, std::vector<float>& out, size_t expected, const char* what) {
  parseFloatLine(lv, out, what);
  if(out.size() != expected)
    throw StringError(
      "Leela Zero weights: line " + Global::intToString(lv.lineNo) + ": " + what + " has " +
      Global::intToString((int)out.size()) + " values, expected " + Global::intToString((int)expected)
    );
}

struct LZConv {
  std::vector<float> w;     //[oc][ic][ky][kx]
  std::vector<float> b;
  std::vector<float> mean;
  std::vector<float> var;
  int lineNo;
};

struct LZNet {
  int formatVersion;
  bool isV2;
  int C;
  int N;
  LZConv ic;
  std::vector<LZConv> blocks;
  LZConv polConv;
  std::vector<float> polFcW;   //[out][in], 362 x 722
  std::vector<float> polFcB;
  LZConv valConv;
  std::vector<float> vhW;     //[out][in], 256 x 361
  std::vector<float> vhB;
  std::vector<float> voW;     //[out][in], 1 x 256
  std::vector<float> voB;
};

LZNet parseLZ(const std::string& data) {
  LineReader r(data);
  LineView lv = nextLineOrThrow(r, "format version");

  int fmt = 0;
  if(!parseFirstInteger(lv.b, lv.e, fmt))
    throw StringError("Leela Zero weights: first line is not an integer format version");

  const int lzOrElf = fmt & 3;
  if(lzOrElf != 1 && lzOrElf != 2)
    throw StringError(
      "Leela Zero weights: format version " + Global::intToString(fmt) +
      ": lz_or_elf (=fmt&3) must be 1 (LZ v1) or 2 (ELF v2)"
    );
  if((fmt & ~3) != 0 || fmt >= 512)
    throw StringError(
      "Leela Zero weights: format version " + Global::intToString(fmt) +
      ": advanced features / chain libraries / chain sizes are not supported (only plain v1/v2 nets)"
    );

  LZNet net;
  net.formatVersion = fmt;
  net.isV2 = (lzOrElf == 2);
  net.C = 0;
  net.N = 0;

  //---- INPUT_CONV: 4 lines ----
  lv = nextLineOrThrow(r, "INPUT_CONV weights");
  const size_t icwLen = countTokens(lv.b, lv.e);
  if(icwLen == 0 || icwLen % (size_t)(LZ_INPUT_PLANES * 9) != 0)
    throw StringError(
      "Leela Zero weights: INPUT_CONV weights has " + Global::intToString((int)icwLen) +
      " values, not a multiple of " + Global::intToString(LZ_INPUT_PLANES * 9) +
      "; nets with extra input planes are not supported"
    );
  const int C = (int)(icwLen / (size_t)(LZ_INPUT_PLANES * 9));
  if(C <= 0)
    throw StringError("Leela Zero weights: INPUT_CONV has zero channels");
  net.C = C;
  parseFloatLine(lv, net.ic.w, (size_t)LZ_INPUT_PLANES * 9 * C, "INPUT_CONV weights");
  lv = nextLineOrThrow(r, "INPUT_CONV bias");
  parseFloatLine(lv, net.ic.b, (size_t)C, "INPUT_CONV bias");
  lv = nextLineOrThrow(r, "INPUT_CONV BN mean");
  parseFloatLine(lv, net.ic.mean, (size_t)C, "INPUT_CONV BN mean");
  lv = nextLineOrThrow(r, "INPUT_CONV BN var");
  parseFloatLine(lv, net.ic.var, (size_t)C, "INPUT_CONV BN var");
  net.ic.lineNo = lv.lineNo;

  //---- RESCONV tower: 2N groups of 4 lines; the tower ends when a line has length C*2 ----
  std::vector<LZConv> groups;
  while(true) {
    lv = nextLineOrThrow(r, "residual/policy tower");
    const size_t len = countTokens(lv.b, lv.e);
    if(len == (size_t)C * 9 * C) {
      LZConv c;
      c.lineNo = lv.lineNo;
      parseFloatLine(lv, c.w, (size_t)C * 9 * C, "RESCONV weights");
      lv = nextLineOrThrow(r, "RESCONV bias");
      parseFloatLine(lv, c.b, (size_t)C, "RESCONV bias");
      lv = nextLineOrThrow(r, "RESCONV BN mean");
      parseFloatLine(lv, c.mean, (size_t)C, "RESCONV BN mean");
      lv = nextLineOrThrow(r, "RESCONV BN var");
      parseFloatLine(lv, c.var, (size_t)C, "RESCONV BN var");
      groups.push_back(std::move(c));
    }
    else {
      if(len != (size_t)C * 2)
        throw StringError(
          "Leela Zero weights: line " + Global::intToString(lv.lineNo) + ": tower line has " +
          Global::intToString((int)len) + " values, expected " +
          Global::intToString(C * 9 * C) + " (RESCONV C*9*C) or " + Global::intToString(C * 2) +
          " (POL_CONV C*2); policy residual towers are not supported"
        );
      LZConv pc;
      pc.lineNo = lv.lineNo;
      parseFloatLine(lv, pc.w, (size_t)C * 2, "POL_CONV weights");
      lv = nextLineOrThrow(r, "POL_CONV bias");
      parseFloatLine(lv, pc.b, 2, "POL_CONV bias (policy outputs must be 2)");
      lv = nextLineOrThrow(r, "POL_CONV BN mean");
      parseFloatLine(lv, pc.mean, 2, "POL_CONV BN mean");
      lv = nextLineOrThrow(r, "POL_CONV BN var");
      parseFloatLine(lv, pc.var, 2, "POL_CONV BN var");
      net.polConv = std::move(pc);
      break;
    }
  }
  if(groups.empty() || groups.size() % 2 != 0)
    throw StringError(
      "Leela Zero weights: residual tower has " + Global::intToString((int)groups.size()) +
      " conv groups, expected a positive even number (conv_a/conv_b pairs)"
    );
  net.N = (int)(groups.size() / 2);
  net.blocks = std::move(groups);

  //---- POL_DENSE: 2 lines, [out][in] ----
  lv = nextLineOrThrow(r, "POL_DENSE weights");
  parseFloatLine(lv, net.polFcW, (size_t)POL_FC_IN * POTENTIAL_MOVES, "POL_DENSE weights");
  lv = nextLineOrThrow(r, "POL_DENSE bias");
  parseFloatLine(lv, net.polFcB, (size_t)POTENTIAL_MOVES, "POL_DENSE bias");

  //---- VALUE_CONV: 4 lines, 1 output ----
  lv = nextLineOrThrow(r, "VALUE_CONV weights");
  const size_t vwLen = countTokens(lv.b, lv.e);
  if(vwLen != (size_t)C)
    throw StringError(
      "Leela Zero weights: VALUE_CONV weights has " + Global::intToString((int)vwLen) +
      " values, expected " + Global::intToString(C) + " (C x 1); value pooling is not supported"
    );
  parseFloatLine(lv, net.valConv.w, (size_t)C, "VALUE_CONV weights");
  net.valConv.lineNo = lv.lineNo;
  lv = nextLineOrThrow(r, "VALUE_CONV bias");
  parseFloatLine(lv, net.valConv.b, 1, "VALUE_CONV bias (value outputs must be 1)");
  lv = nextLineOrThrow(r, "VALUE_CONV BN mean");
  parseFloatLine(lv, net.valConv.mean, 1, "VALUE_CONV BN mean");
  lv = nextLineOrThrow(r, "VALUE_CONV BN var");
  parseFloatLine(lv, net.valConv.var, 1, "VALUE_CONV BN var");

  //---- VAL_DENSE_HIDDEN: 2 lines ----
  lv = nextLineOrThrow(r, "VAL_DENSE_HIDDEN weights");
  parseFloatLine(lv, net.vhW, (size_t)NUM_INTERSECTIONS * VALUE_HIDDEN, "VAL_DENSE_HIDDEN weights");
  lv = nextLineOrThrow(r, "VAL_DENSE_HIDDEN bias");
  parseFloatLine(lv, net.vhB, (size_t)VALUE_HIDDEN, "VAL_DENSE_HIDDEN bias");

  //---- VAL_DENSE_OUT: 2 lines, single output (SINGLE head) ----
  lv = nextLineOrThrow(r, "VAL_DENSE_OUT weights");
  parseFloatLine(lv, net.voW, (size_t)VALUE_HIDDEN, "VAL_DENSE_OUT weights");
  lv = nextLineOrThrow(r, "VAL_DENSE_OUT bias");
  parseFloatLine(lv, net.voB, 1, "VAL_DENSE_OUT bias (value head must be SINGLE)");

  //---- EOF: any trailing data means a nonstandard topology (VBE / double head) ----
  if(r.nextLine(lv.b, lv.e))
    throw StringError(
      "Leela Zero weights: line " + Global::intToString(r.lineNo) +
      ": unexpected trailing data (VBE / double value head not supported)"
    );

  return net;
}

//Folds an LZ batch norm into (scale, bias) for KataGo's canonical BN (mean=0, var=1).
void foldBN(const std::vector<float>& b, const std::vector<float>& mean, const std::vector<float>& var, int n, std::vector<float>& scale, std::vector<float>& bias) {
  scale.resize(n);
  bias.resize(n);
  for(int i = 0; i < n; i++) {
    const float s = 1.0f / std::sqrt(var[i] + LZ_BN_EPS);
    scale[i] = s;
    bias[i] = s * (b[i] - mean[i]);
  }
}

std::vector<float> ones(int n) {
  return std::vector<float>((size_t)n, 1.0f);
}
std::vector<float> zeros(int n) {
  return std::vector<float>((size_t)n, 0.0f);
}

void initBN(BatchNormLayerDesc& bn, const std::string& name, int n, std::vector<float>&& scale, std::vector<float>&& bias) {
  bn.name = name;
  bn.numChannels = n;
  bn.epsilon = KG_BN_EPS;
  bn.hasScale = true;
  bn.hasBias = true;
  bn.mean.assign((size_t)n, 0.0f);
  bn.variance.assign((size_t)n, 1.0f);
  bn.scale = std::move(scale);
  bn.bias = std::move(bias);
  bn.computeMerged();
}

void initConv(ConvLayerDesc& c, const std::string& name, int ky, int kx, int ic, int oc, std::vector<float>&& w) {
  c.name = name;
  c.convYSize = ky;
  c.convXSize = kx;
  c.inChannels = ic;
  c.outChannels = oc;
  c.dilationY = 1;
  c.dilationX = 1;
  c.weights = std::move(w);
}

void initMatMul(MatMulLayerDesc& m, const std::string& name, int in, int out, std::vector<float>&& w) {
  m.name = name;
  m.inChannels = in;
  m.outChannels = out;
  m.weights = std::move(w);
}

void initMatBias(MatBiasLayerDesc& b, const std::string& name, int n, std::vector<float>&& w) {
  b.name = name;
  b.numChannels = n;
  b.weights = std::move(w);
}

void initActivation(ActivationLayerDesc& a, const std::string& name, int activation) {
  a.name = name;
  a.activation = activation;
}

//Builds one KataGo postnorm residual block from a pair of LZ conv+BN groups.
//For the dummy first block (index 0) the residual branch is entirely zero and the post BN
//applies the input convolution's folded bias+BN.
void buildBlock(
  TrunkDesc& trunk,
  int index,
  int C,
  const LZConv* a,
  const LZConv* b,
  const std::vector<float>& postScale,
  const std::vector<float>& postBias
) {
  const std::string n = Global::intToString(index + 1);
  ResidualBlockDesc* blk = new ResidualBlockDesc();
  blk->name = "block" + n;
  blk->hasPostNorm = true;

  //preBN / preAct are identity: the input is already nonnegative (previous postAct was ReLU).
  initBN(blk->preBN, "preBN" + n, C, ones(C), zeros(C));
  initActivation(blk->preActivation, "preAct" + n, ACTIVATION_IDENTITY);

  if(a == NULL) {
    //Dummy block: 1x1 zero convolutions, zero BN, so the residual branch is identically zero.
    initConv(blk->regularConv, "regularConv" + n, 1, 1, C, C, zeros(C * C));
    initBN(blk->midBN, "midBN" + n, C, ones(C), zeros(C));
    initActivation(blk->midActivation, "midAct" + n, ACTIVATION_RELU);
    initConv(blk->finalConv, "finalConv" + n, 1, 1, C, C, zeros(C * C));
  }
  else {
    std::vector<float> aScale, aBias, bScale, bBias;
    foldBN(a->b, a->mean, a->var, C, aScale, aBias);
    foldBN(b->b, b->mean, b->var, C, bScale, bBias);

    initConv(blk->regularConv, "regularConv" + n, 3, 3, C, C, std::vector<float>(a->w));
    initBN(blk->midBN, "midBN" + n, C, std::move(aScale), std::move(aBias));
    initActivation(blk->midActivation, "midAct" + n, ACTIVATION_RELU);

    //The second BN's scale is folded into the output channels of the final conv; its bias goes
    //into the (post-residual) post BN, matching LZ's x + BN_b(conv_b(...)) then ReLU.
    std::vector<float> finW(b->w);
    for(int oc = 0; oc < C; oc++)
      for(int i = 0; i < C * 9; i++)
        finW[(size_t)oc * C * 9 + i] *= bScale[oc];
    initConv(blk->finalConv, "finalConv" + n, 3, 3, C, C, std::move(finW));
    initBN(blk->postBN, "postBN" + n, C, ones(C), std::move(bBias));
    initActivation(blk->postActivation, "postAct" + n, ACTIVATION_RELU);
    trunk.blocks.emplace_back(POSTNORM_BLOCK_KIND, make_unique_void(blk));
    return;
  }

  initBN(blk->postBN, "postBN" + n, C, std::vector<float>(postScale), std::vector<float>(postBias));
  initActivation(blk->postActivation, "postAct" + n, ACTIVATION_RELU);
  trunk.blocks.emplace_back(POSTNORM_BLOCK_KIND, make_unique_void(blk));
}

void fillModelDesc(const LZNet& net, const std::string& sha256, ModelDesc& desc) {
  const int C = net.C;
  const int N = net.N;

  //---- ModelDesc header ----
  desc = ModelDesc();
  desc.name = "lz_v" + Global::intToString(net.isV2 ? 2 : 1) + "_c" + Global::intToString(C) + "_b" + Global::intToString(N);
  desc.sha256 = sha256;
  desc.modelVersion = 18;   //V8 inputs; LZ identity is carried by isLZModel
  desc.isLZModel = true;
  desc.numInputChannels = 19;
  desc.numInputGlobalChannels = 1;
  desc.numInputMetaChannels = 0;
  desc.metaEncoderVersion = 0;
  desc.preferPassAliveUnderSuicideRules = false;
  desc.preferExcludeTerritoryAdjacentToAtari = false;
  desc.valueFromBlackPerspective = net.isV2;

  //The generated model text historically carried all-ones multipliers, keep them identical.
  desc.postProcessParams.tdScoreMultiplier = 1.0;
  desc.postProcessParams.scoreMeanMultiplier = 1.0;
  desc.postProcessParams.scoreStdevMultiplier = 1.0;
  desc.postProcessParams.leadMultiplier = 1.0;
  desc.postProcessParams.varianceTimeMultiplier = 1.0;
  desc.postProcessParams.shorttermValueErrorMultiplier = 1.0;
  desc.postProcessParams.shorttermScoreErrorMultiplier = 1.0;
  desc.postProcessParams.outputScaleMultiplier = 1.0f;

  desc.numPolicyChannels = 1;
  desc.numValueChannels = 3;
  desc.numScoreValueChannels = 6;
  desc.numOwnershipChannels = 1;

  //---- Trunk ----
  TrunkDesc& trunk = desc.trunk;
  trunk.name = "trunk";
  trunk.modelVersion = 18;
  trunk.numBlocks = N + 1;   //dummy block + N postnorm blocks
  trunk.trunkNumChannels = C;
  trunk.midNumChannels = C;
  trunk.regularNumChannels = C;
  trunk.gpoolNumChannels = C;
  trunk.metaEncoderVersion = 0;
  trunk.trunkNormKind = TRUNK_NORM_KIND_STANDARD;

  //LZ has 18 planes -> V8 channels 1..18; channel 0 (on-board mask) gets zero weights.
  std::vector<float> icW((size_t)C * 19 * 9, 0.0f);
  for(int oc = 0; oc < C; oc++)
    for(int ic = 0; ic < LZ_INPUT_PLANES; ic++)
      for(int k = 0; k < 9; k++)
        icW[(size_t)oc * 19 * 9 + (size_t)(ic + 1) * 9 + k] = net.ic.w[(size_t)oc * LZ_INPUT_PLANES * 9 + (size_t)ic * 9 + k];
  initConv(trunk.initialConv, "initialConv", 3, 3, 19, C, std::move(icW));
  initMatMul(trunk.initialMatMul, "initialMatMul", 1, C, zeros(C));

  std::vector<float> icScale, icBias;
  foldBN(net.ic.b, net.ic.mean, net.ic.var, C, icScale, icBias);
  buildBlock(trunk, 0, C, NULL, NULL, icScale, icBias);
  for(int i = 0; i < N; i++) {
    const LZConv& a = net.blocks[(size_t)2 * i];
    const LZConv& b = net.blocks[(size_t)2 * i + 1];
    buildBlock(trunk, i + 1, C, &a, &b, std::vector<float>(), std::vector<float>());
  }

  initBN(trunk.trunkTipBN, "trunkTipBN", C, ones(C), zeros(C));
  initActivation(trunk.trunkTipActivation, "trunkTipActivation", ACTIVATION_IDENTITY);

  //---- Policy head ----
  PolicyHeadDesc& pol = desc.policyHead;
  pol.name = "policy";
  pol.modelVersion = 18;
  pol.policyOutChannels = 1;
  pol.hasFullPolicyMul = true;

  initConv(pol.p1Conv, "p1Conv", 1, 1, C, 2, std::vector<float>(net.polConv.w));

  std::vector<float> polScale, polBias;
  foldBN(net.polConv.b, net.polConv.mean, net.polConv.var, 2, polScale, polBias);

  //Unused global-pooling policy path: dimensionally valid all-zero dummies.
  initConv(pol.g1Conv, "g1Conv", 1, 1, C, 1, zeros(C));
  initBN(pol.g1BN, "g1BN", 1, ones(1), zeros(1));
  initActivation(pol.g1Activation, "g1Activation", ACTIVATION_RELU);
  initMatMul(pol.gpoolToBiasMul, "gpoolToBiasMul", 3, 2, zeros(6));

  initBN(pol.p1BN, "p1BN", 2, std::move(polScale), std::move(polBias));
  //LZ's policy path uses the batchnorm template, which clamps at zero (CPUPipe.h: the
  //eltwise==nullptr branch does std::max(0.0f, ...)); the policy FC itself passes ReLU=false.
  initActivation(pol.p1Activation, "p1Activation", ACTIVATION_RELU);

  //LZ file order is [out=362][in=722] with input index c*361+pos; KataGo's flattened input is
  //position-major pos*2+c, and the engine's output index j=y*19+x maps to LZ position
  //flip(j)=(18-y)*19+x (pass index 361 stays).
  std::vector<float> pfullW((size_t)POL_FC_IN * POTENTIAL_MOVES);
  for(int pos = 0; pos < NUM_INTERSECTIONS; pos++) {
    for(int c = 0; c < 2; c++) {
      const int inIdx = pos * 2 + c;
      const int lzIn = c * NUM_INTERSECTIONS + pos;
      for(int j = 0; j < POTENTIAL_MOVES; j++) {
        const int lzOut = (j < NUM_INTERSECTIONS)
          ? ((BOARD_SIZE - 1 - (j / BOARD_SIZE)) * BOARD_SIZE + (j % BOARD_SIZE))
          : NUM_INTERSECTIONS;
        pfullW[(size_t)inIdx * POTENTIAL_MOVES + j] = net.polFcW[(size_t)lzOut * POL_FC_IN + lzIn];
      }
    }
  }
  initMatMul(pol.pFullMul, "pFullMul", POL_FC_IN, POTENTIAL_MOVES, std::move(pfullW));

  std::vector<float> pfullB((size_t)POTENTIAL_MOVES);
  for(int j = 0; j < POTENTIAL_MOVES; j++) {
    const int lzOut = (j < NUM_INTERSECTIONS)
      ? ((BOARD_SIZE - 1 - (j / BOARD_SIZE)) * BOARD_SIZE + (j % BOARD_SIZE))
      : NUM_INTERSECTIONS;
    pfullB[j] = net.polFcB[lzOut];
  }
  initMatBias(pol.pFullMulBias, "pFullMulBias", POTENTIAL_MOVES, std::move(pfullB));

  initConv(pol.p2Conv, "p2Conv", 1, 1, 2, 1, zeros(2));
  initMatMul(pol.gpoolToPassMul, "gpoolToPassMul", 3, 2, zeros(6));
  initMatBias(pol.gpoolToPassBias, "gpoolToPassBias", 2, zeros(2));
  initActivation(pol.passActivation, "passActivation", ACTIVATION_RELU);
  initMatMul(pol.gpoolToPassMul2, "gpoolToPassMul2", 2, 1, zeros(2));

  //---- Value head ----
  ValueHeadDesc& val = desc.valueHead;
  val.name = "value";
  val.modelVersion = 18;
  val.hasFullValueMul = true;

  initConv(val.v1Conv, "v1Conv", 1, 1, C, 1, std::vector<float>(net.valConv.w));

  std::vector<float> valScale, valBias;
  foldBN(net.valConv.b, net.valConv.mean, net.valConv.var, 1, valScale, valBias);
  initBN(val.v1BN, "v1BN", 1, std::move(valScale), std::move(valBias));
  initActivation(val.v1Activation, "v1Activation", ACTIVATION_RELU);

  //C=1: LZ channel-major and KataGo position-major flattenings coincide, only transpose.
  std::vector<float> v2W((size_t)NUM_INTERSECTIONS * VALUE_HIDDEN);
  for(int pos = 0; pos < NUM_INTERSECTIONS; pos++)
    for(int h = 0; h < VALUE_HIDDEN; h++)
      v2W[(size_t)pos * VALUE_HIDDEN + h] = net.vhW[(size_t)h * NUM_INTERSECTIONS + pos];
  initMatMul(val.v2Mul, "v2Mul", NUM_INTERSECTIONS, VALUE_HIDDEN, std::move(v2W));
  initMatBias(val.v2Bias, "v2Bias", VALUE_HIDDEN, std::vector<float>(net.vhB));
  initActivation(val.v2Activation, "v2Activation", ACTIVATION_RELU);

  //Single-valued LZ value head -> KataGo's (win, loss, draw) triple. The draw logit -30 makes
  //draw probability numerically zero, matching LZ's sigmoid(2z) win probability.
  std::vector<float> v3W((size_t)VALUE_HIDDEN * 3);
  for(int h = 0; h < VALUE_HIDDEN; h++) {
    const float w2 = net.voW[(size_t)h];
    v3W[(size_t)h * 3 + 0] = w2;
    v3W[(size_t)h * 3 + 1] = -w2;
    v3W[(size_t)h * 3 + 2] = 0.0f;
  }
  initMatMul(val.v3Mul, "v3Mul", VALUE_HIDDEN, 3, std::move(v3W));

  const float b2 = net.voB[0];
  std::vector<float> v3B(3);
  v3B[0] = b2;
  v3B[1] = -b2;
  v3B[2] = -30.0f;
  initMatBias(val.v3Bias, "v3Bias", 3, std::move(v3B));

  //Dummy score and ownership heads: LZ nets have neither, so the main net reports 0.
  initMatMul(val.sv3Mul, "sv3Mul", VALUE_HIDDEN, 6, zeros(VALUE_HIDDEN * 6));
  initMatBias(val.sv3Bias, "sv3Bias", 6, zeros(6));
  initConv(val.vOwnershipConv, "vOwnershipConv", 1, 1, 1, 1, zeros(1));
}

}  // namespace

namespace LZConverter {

bool looksLikeLZ(const std::string& data) {
  LineReader r(data);
  const char* b;
  const char* e;
  if(!r.nextLine(b, e)) return false;

  int fmt = 0;
  if(!parseFirstInteger(b, e, fmt)) return false;
  const int lzOrElf = fmt & 3;
  if(lzOrElf != 1 && lzOrElf != 2) return false;
  if((fmt & ~3) != 0 || fmt >= 512) return false;

  if(!r.nextLine(b, e)) return false;
  const size_t icwLen = countTokens(b, e);
  if(icwLen == 0 || icwLen % (size_t)(LZ_INPUT_PLANES * 9) != 0) return false;
  const size_t C = icwLen / (size_t)(LZ_INPUT_PLANES * 9);
  if(C < 32 || C > 1024) return false;

  if(!r.nextLine(b, e) || countTokens(b, e) != C) return false;          //bias
  if(!r.nextLine(b, e) || countTokens(b, e) != C) return false;          //BN mean
  if(!r.nextLine(b, e) || countTokens(b, e) != C) return false;          //BN var
  if(!r.nextLine(b, e) || countTokens(b, e) != C * 9 * C) return false;  //first residual conv
  return true;
}

void buildModelDesc(const std::string& data, const std::string& sha256, ModelDesc& desc) {
  const LZNet net = parseLZ(data);
  fillModelDesc(net, sha256, desc);
}

}  // namespace LZConverter
