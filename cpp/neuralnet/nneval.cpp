#include "../neuralnet/nneval.h"
#include "../neuralnet/modelversion.h"
#include "../core/test.h"

#include <algorithm>
#include <chrono>

using namespace std;

//-------------------------------------------------------------------------------------

NNResultBuf::NNResultBuf()
  : clientWaitingForResult(),
    resultMutex(),
    hasResult(false),
    includeOwnerMap(false),
    boardXSizeForServer(0),
    boardYSizeForServer(0),
    rowSpatialBuf(),
    rowGlobalBuf(),
    rowMetaBuf(),
    hasRowMeta(false),
    result(nullptr),
    errorLogLockout(false),
    // If no symmetry is specified, it will use default or random based on config.
    symmetry(NNInputs::SYMMETRY_NOTSPECIFIED),
    policyOptimism(0.0)
{}

NNResultBuf::~NNResultBuf() {
}

//-------------------------------------------------------------------------------------

NNServerBuf::NNServerBuf(const NNEvaluator& nnEval, const LoadedModel* model)
  :inputBuffers(NULL)
{
  int maxBatchSize = nnEval.getMaxBatchSize();
  if(model != NULL)
    inputBuffers = NeuralNet::createInputBuffers(model,maxBatchSize,nnEval.getNNXLen(),nnEval.getNNYLen());
}

NNServerBuf::~NNServerBuf() {
  if(inputBuffers != NULL)
    NeuralNet::freeInputBuffers(inputBuffers);
  inputBuffers = NULL;
}

//-------------------------------------------------------------------------------------

NNEvaluator::NNEvaluator(
  const string& mName,
  const string& mFileName,
  const string& expectedSha256,
  Logger* lg,
  int maxBatchSz,
  int xLen,
  int yLen,
  bool rExactNNLen,
  bool iUseNHWC,
  int nnCacheSizePowerOfTwo,
  int nnMutexPoolSizePowerofTwo,
  bool skipNeuralNet,
  const string& homeDataDirOverride,
  enabled_t useFP16Mode,
  int numThr,
  const vector<int>& gpuIdxByServerThr,
  const string& rSeed,
  bool doRandomize,
  int defaultSymmetry,
  ConfigParser& cfg
)
  :modelName(mName),
   modelFileName(mFileName),
   nnXLen(xLen),
   nnYLen(yLen),
   requireExactNNLen(rExactNNLen),
   policySize(NNPos::getPolicySize(xLen,yLen)),
   inputsUseNHWC(iUseNHWC),
   usingFP16Mode(useFP16Mode),
   numThreads(numThr),
   gpuIdxByServerThread(gpuIdxByServerThr),
   randSeed(rSeed),
   debugSkipNeuralNet(skipNeuralNet),
   computeContext(NULL),
   loadedModel(NULL),
   nnCacheTable(NULL),
   logger(lg),
   internalModelName(),
   modelVersion(-1),
   isLZModelFlag(false),
   inputsVersion(-1),
   valueFromBlackPerspective(false),
   numInputMetaChannels(0),
   postProcessParams(),
   numServerThreadsEverSpawned(0),
   serverThreads(),
   maxBatchSize(maxBatchSz),
   numEffectiveDevices(NeuralNet::getNumEffectiveDevices(cfg, gpuIdxByServerThr)),
   m_numRowsProcessed(0),
   m_numBatchesProcessed(0),
   m_numCacheHits(0),
   bufferMutex(),
   isKilled(false),
   numServerThreadsStartingUp(0),
   mainThreadWaitingForSpawn(),
   numOngoingEvals(0),
   numWaitingEvals(0),
   numEvalsToAwaken(0),
   waitingForFinish(),
   currentDoRandomize(doRandomize),
   currentDefaultSymmetry(defaultSymmetry),
   maxRowsToSendPerBatch(maxBatchSz),
   queryQueue()
{
  if(nnXLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(nnYLen > NNPos::MAX_BOARD_LEN)
    throw StringError("Maximum supported nnEval board size is " + Global::intToString(NNPos::MAX_BOARD_LEN));
  if(maxBatchSize <= 0)
    throw StringError("maxBatchSize is negative: " + Global::intToString(maxBatchSize));
  if(gpuIdxByServerThread.size() != numThreads)
    throw StringError("gpuIdxByServerThread.size() != numThreads");

  if(logger != NULL) {
    logger->write(
      "Initializing neural net buffer to be size " +
      Global::intToString(nnXLen) + " * " + Global::intToString(nnYLen) +
      (requireExactNNLen ? " exactly" : " allowing smaller boards")
    );
  }

  if(nnCacheSizePowerOfTwo >= 0)
    nnCacheTable = new NNCacheTable(nnCacheSizePowerOfTwo, nnMutexPoolSizePowerofTwo);

  if(!debugSkipNeuralNet) {
    vector<int> gpuIdxs = gpuIdxByServerThread;
    std::sort(gpuIdxs.begin(), gpuIdxs.end());
    auto last = std::unique(gpuIdxs.begin(), gpuIdxs.end());
    gpuIdxs.erase(last,gpuIdxs.end());
    loadedModel = NeuralNet::loadModelFile(modelFileName,expectedSha256);
    const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
    internalModelName = desc.name;
    modelVersion = desc.modelVersion;
    isLZModelFlag = desc.isLZModel;
    //Leela Zero-style networks have their spatial size fixed at 19x19.
    if(isLZModelFlag && (nnXLen != NNPos::MAX_BOARD_LEN || nnYLen != NNPos::MAX_BOARD_LEN))
      throw StringError(
        "Leela Zero model requires a " +
        Global::intToString(NNPos::MAX_BOARD_LEN) + " x " + Global::intToString(NNPos::MAX_BOARD_LEN) +
        " neural net buffer, but the configured buffer is " +
        Global::intToString(nnXLen) + " x " + Global::intToString(nnYLen));
    //LZ nets use the V8 input layout: 19 spatial planes (an on-board mask plus the 18 LZ
    //history/color planes) and a single dummy global channel, with no meta encoder. Assert it here
    //so that a model claiming to be LZ but carrying some other layout fails loudly instead of
    //being silently featurized the wrong way.
    if(isLZModelFlag) {
      if(desc.numInputChannels != NNInputs::NUM_FEATURES_SPATIAL_V8 ||
         desc.numInputGlobalChannels != NNInputs::NUM_FEATURES_GLOBAL_V8 ||
         desc.metaEncoderVersion != 0)
        throw StringError(
          "Leela Zero model has an unexpected input layout: " +
          Global::intToString(desc.numInputChannels) + " spatial channels (expected " +
          Global::intToString(NNInputs::NUM_FEATURES_SPATIAL_V8) + "), " +
          Global::intToString(desc.numInputGlobalChannels) + " global channels (expected " +
          Global::intToString(NNInputs::NUM_FEATURES_GLOBAL_V8) + "), metaEncoderVersion " +
          Global::intToString(desc.metaEncoderVersion) + " (expected 0)");
    }
    valueFromBlackPerspective = desc.valueFromBlackPerspective;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
    numInputMetaChannels = desc.numInputMetaChannels;
    computeContext = NeuralNet::createComputeContext(
      gpuIdxs,logger,nnXLen,nnYLen,
      homeDataDirOverride,
      usingFP16Mode,loadedModel,cfg
    );
    // Snapshot postProcessParams only after createComputeContext: backends may apply
    // config-dependent transforms to the model desc there (e.g. the ONNX backend's
    // scale8 workaround multiplies outputScaleMultiplier by 8).
    postProcessParams = desc.postProcessParams;
  }
  else {
    internalModelName = "random";
    modelVersion = NNModelVersion::defaultModelVersion;
    isLZModelFlag = false;
    inputsVersion = NNModelVersion::getInputsVersion(modelVersion);
  }

  // Reserve a decent amount above the batch size so that allocation is unlikely.
  queryQueue.reserve(maxBatchSize * 4 * gpuIdxByServerThread.size());
  // Starts readonly. Becomes writable once we spawn server threads
  queryQueue.setReadOnly();
}

NNEvaluator::~NNEvaluator() {
  killServerThreads();

  if(computeContext != NULL)
    NeuralNet::freeComputeContext(computeContext);
  computeContext = NULL;

  if(loadedModel != NULL)
    NeuralNet::freeLoadedModel(loadedModel);
  loadedModel = NULL;

  delete nnCacheTable;
}

string NNEvaluator::getModelName() const {
  return modelName;
}
string NNEvaluator::getModelFileName() const {
  return modelFileName;
}
string NNEvaluator::getInternalModelName() const {
  return internalModelName;
}

static bool tryAbbreviateStepString(const string& input, string& buf) {
  size_t i = 0;
  while(i < input.length() && !Global::isDigit(input[i]))
    i++;
  if(i > 1)
    return false;

  string prefix = input.substr(0, i);
  int64_t number;
  bool suc = Global::tryStringToInt64(input.substr(i),number);
  if(!suc)
    return false;

  if(number >= 10000000000LL)
    buf = prefix + std::to_string(number / 1000000000LL) + "G";
  if(number >= 10000000)
    buf = prefix + std::to_string(number / 1000000) + "M";
  else if(number >= 10000)
    buf = prefix + std::to_string(number / 1000) + "K";
  else
    buf = input;
  return true;
}

string NNEvaluator::getAbbrevInternalModelName() const {
  string name = getInternalModelName();
  std::vector<string> pieces = Global::split(name,'-');
  std::vector<string> newPieces;
  for(const string& piece: pieces) {
    string buf;
    if(piece == "kata1") {
      // skip
    }
    else if(piece.size() > 1 && piece[0] == 's' && tryAbbreviateStepString(piece,buf)) {
      newPieces.push_back(buf);
    }
    else if(piece.size() > 1 && piece[0] == 'd' && tryAbbreviateStepString(piece,buf)) {
      // skip
    }
    else {
      newPieces.push_back(piece);
    }
  }
  return Global::concat(newPieces,"-");
}

Logger* NNEvaluator::getLogger() {
  return logger;
}
bool NNEvaluator::isNeuralNetLess() const {
  return debugSkipNeuralNet;
}
int NNEvaluator::getMaxBatchSize() const {
  return maxBatchSize;
}
int NNEvaluator::getMaxRowsToSendPerBatch() const {
  return maxRowsToSendPerBatch.load(std::memory_order_acquire);
}
void NNEvaluator::setMaxRowsToSendPerBatch(int maxRows) {
  if(maxRows <= 0 || maxRows > maxBatchSize)
    throw StringError("Invalid setting for max rows to send per batch");
  maxRowsToSendPerBatch.store(maxRows,std::memory_order_release);
}
bool NNEvaluator::requiresSGFMetadata() const {
  return numInputMetaChannels > 0;
}

int NNEvaluator::getNumGpus() const {
  return numEffectiveDevices;
}
int NNEvaluator::getNumServerThreads() const {
  return (int)gpuIdxByServerThread.size();
}
const std::vector<int>& NNEvaluator::getGpuIdxByServerThread() const {
  return gpuIdxByServerThread;
}
std::set<int> NNEvaluator::getGpuIdxs() const {
  std::set<int> gpuIdxs;
#ifdef USE_EIGEN_BACKEND
  gpuIdxs.insert(0);
#else
  for(int i = 0; i<gpuIdxByServerThread.size(); i++) {
    gpuIdxs.insert(gpuIdxByServerThread[i]);
  }
#endif
  return gpuIdxs;
}

int NNEvaluator::getNNXLen() const {
  return nnXLen;
}
int NNEvaluator::getNNYLen() const {
  return nnYLen;
}
bool NNEvaluator::getRequireExactNNLen() const {
  return requireExactNNLen;
}
int NNEvaluator::getModelVersion() const {
  return modelVersion;
}
bool NNEvaluator::isLZModel() const {
  return isLZModelFlag;
}
double NNEvaluator::getTrunkSpatialConvDepth() const {
  return NeuralNet::getModelDesc(loadedModel).getTrunkSpatialConvDepth();
}

int64_t NNEvaluator::getNumModelParameters() const {
  return NeuralNet::getModelDesc(loadedModel).getNumParameters();
}

bool NNEvaluator::modelHasAnyTransformerBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks();
}

bool NNEvaluator::modelHasAnyNestedBottleneckBlocks() const {
  return NeuralNet::getModelDesc(loadedModel).hasAnyNestedBottleneckBlocks();
}

enabled_t NNEvaluator::getUsingFP16Mode() const {
  return usingFP16Mode;
}

bool NNEvaluator::supportsShorttermError() const {
  //Leela-Zero-style nets have no shortterm error outputs. Keyed on the explicit LZ marker rather
  //than on a version bound, so that a future upstream version raising the bound does not silently
  //change what an LZ net is taken to support.
  if(isLZModelFlag)
    return false;
  return modelVersion >= 9;
}

bool NNEvaluator::modelPreferPassAliveUnderSuicideRules() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferPassAliveUnderSuicideRules;
}

bool NNEvaluator::modelPreferExcludeTerritoryAdjacentToAtari() const {
  if(loadedModel == NULL)
    return false;
  return NeuralNet::getModelDesc(loadedModel).preferExcludeTerritoryAdjacentToAtari;
}

bool NNEvaluator::getDoRandomize() const {
  return currentDoRandomize.load(std::memory_order_acquire);
}
int NNEvaluator::getDefaultSymmetry() const {
  return currentDefaultSymmetry.load(std::memory_order_acquire);
}
void NNEvaluator::setDoRandomize(bool b) {
  currentDoRandomize.store(b, std::memory_order_release);
}
void NNEvaluator::setDefaultSymmetry(int s) {
  currentDefaultSymmetry.store(s, std::memory_order_release);
}

Rules NNEvaluator::getSupportedRules(const Rules& desiredRules, bool& supported) const {
  if(loadedModel == NULL) {
    supported = true;
    return desiredRules;
  }
  return NeuralNet::getModelDesc(loadedModel).getSupportedRules(desiredRules, supported);
}

uint64_t NNEvaluator::numRowsProcessed() const {
  return m_numRowsProcessed.load(std::memory_order_relaxed);
}
uint64_t NNEvaluator::numBatchesProcessed() const {
  return m_numBatchesProcessed.load(std::memory_order_relaxed);
}
double NNEvaluator::averageProcessedBatchSize() const {
  return (double)numRowsProcessed() / (double)numBatchesProcessed();
}
uint64_t NNEvaluator::numCacheHits() const {
  return m_numCacheHits.load(std::memory_order_relaxed);
}

void NNEvaluator::clearStats() {
  m_numRowsProcessed.store(0);
  m_numBatchesProcessed.store(0);
  m_numCacheHits.store(0);
}

void NNEvaluator::clearCache() {
  if(nnCacheTable != NULL)
    nnCacheTable->clear();
}


bool NNEvaluator::isAnyThreadUsingFP16() const {
  lock_guard<std::mutex> lock(bufferMutex);
  for(const int& isUsingFP16: serverThreadsIsUsingFP16) {
    if(isUsingFP16)
      return true;
  }
  return false;
}

static void serveEvals(
  string randSeedThisThread,
  NNEvaluator* nnEval,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  Rand rand(randSeedThisThread);

  // We're in big trouble if this raises an exception (e.g. out of GPU memory) and causes possibly
  // the only nnEval thread to die, so let the exception escape and terminate the process. But log
  // and print the error first, since on some platforms an exception escaping a thread aborts
  // without printing anything. cerr is used unconditionally rather than only when there is no
  // logger, since the logger may be temporarily disabled (e.g. the benchmark suppresses logging
  // while respawning server threads).
  try {
    nnEval->serve(rand,gpuIdxForThisThread,serverThreadIdx);
  }
  catch(const std::exception& e) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write(string("ERROR: NN server thread failed: ") + e.what());
    cerr << (string("ERROR: NN server thread failed: ") + e.what()) << endl;
    throw;
  }
  catch(...) {
    Logger* logger = nnEval->getLogger();
    if(logger != NULL)
      logger->write("ERROR: NN server thread failed with an exception that is not a std::exception");
    cerr << "ERROR: NN server thread failed with an exception that is not a std::exception" << endl;
    throw;
  }
}

void NNEvaluator::setNumThreads(const vector<int>& gpuIdxByServerThr) {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::setNumThreads called when threads were already running!");
  numThreads = (int)gpuIdxByServerThr.size();
  gpuIdxByServerThread = gpuIdxByServerThr;
}

void NNEvaluator::spawnServerThreads() {
  if(serverThreads.size() != 0)
    throw StringError("NNEvaluator::spawnServerThreads called when threads were already running!");

  {
    lock_guard<std::mutex> lock(bufferMutex);
    serverThreadsIsUsingFP16.resize(numThreads,0);
  }

  queryQueue.unsetReadOnly();

  numServerThreadsStartingUp = numThreads;
  for(int i = 0; i<numThreads; i++) {
    int gpuIdxForThisThread = gpuIdxByServerThread[i];
    string randSeedThisThread = randSeed + ":NNEvalServerThread:" + Global::intToString(numServerThreadsEverSpawned);
    numServerThreadsEverSpawned++;
    std::thread* thread = new std::thread(
      &serveEvals,randSeedThisThread,this,gpuIdxForThisThread,i
    );
    serverThreads.push_back(thread);
  }

  unique_lock<std::mutex> lock(bufferMutex);
  while(numServerThreadsStartingUp > 0)
    mainThreadWaitingForSpawn.wait(lock);
}

void NNEvaluator::killServerThreads() {
  unique_lock<std::mutex> lock(bufferMutex);
  isKilled = true;
  lock.unlock();
  queryQueue.setReadOnly();

  waitingForFinish.notify_all();

  for(size_t i = 0; i<serverThreads.size(); i++)
    serverThreads[i]->join();
  for(size_t i = 0; i<serverThreads.size(); i++)
    delete serverThreads[i];
  serverThreads.clear();
  serverThreadsIsUsingFP16.clear();

  // Can unset now that threads are dead
  isKilled = false;

  testAssert(numOngoingEvals == 0);
  testAssert(numWaitingEvals == 0);
  testAssert(numEvalsToAwaken == 0);
}

void NNEvaluator::fillRowBufs(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf
) const {
  const int rowSpatialLen = NNModelVersion::getNumSpatialFeatures(modelVersion) * nnXLen * nnYLen;
  if(buf.rowSpatialBuf.size() < rowSpatialLen)
    buf.rowSpatialBuf.resize(rowSpatialLen);
  const int rowGlobalLen = NNModelVersion::getNumGlobalFeatures(modelVersion);
  if(buf.rowGlobalBuf.size() < rowGlobalLen)
    buf.rowGlobalBuf.resize(rowGlobalLen);
  const int rowMetaLen = numInputMetaChannels;
  if(buf.rowMetaBuf.size() < rowMetaLen)
    buf.rowMetaBuf.resize(rowMetaLen);

  static_assert(NNModelVersion::latestInputsVersionImplemented == 8, "");
  if(inputsVersion == 3)
    NNInputs::fillRowV3(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 4)
    NNInputs::fillRowV4(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 5)
    NNInputs::fillRowV5(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 6)
    NNInputs::fillRowV6(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 7)
    NNInputs::fillRowV7(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else if(inputsVersion == 8)
    NNInputs::fillRowV8(board, history, nextPlayer, nnInputParams, nnXLen, nnYLen, inputsUseNHWC, buf.rowSpatialBuf.data(), buf.rowGlobalBuf.data());
  else
    ASSERT_UNREACHABLE;

  if(rowMetaLen > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    SGFMetadata::fillMetadataRow(
      sgfMeta,
      buf.rowMetaBuf.data(),
      nextPlayer,
      board.x_size*board.y_size
    );
    buf.hasRowMeta = true;
  }
  else {
    buf.hasRowMeta = false;
  }
}

void NNEvaluator::maybeWarmupComputeHandle(ComputeHandle* gpuHandle, int serverThreadIdx) {
  if(gpuHandle == NULL || debugSkipNeuralNet || loadedModel == NULL)
    return;
  // Warmup currently only matters on CUDA, where cuDNN lazily compiles an SDPA execution plan per
  // batch size on first use. Other backends: nothing to warm up for now.
#if !defined(USE_CUDA_BACKEND)
  (void)serverThreadIdx;
  return;
#else
  // Only transformer models build the lazy SDPA graphs; skip the (otherwise harmless but wasteful)
  // warmup passes for plain convnets.
  if(!NeuralNet::getModelDesc(loadedModel).hasAnyTransformerBlocks())
    return;

  if(logger != NULL) {
    logger->write(
      "Cuda backend thread " + Global::intToString(serverThreadIdx) +
      ": warming up transformer graphs for batch sizes 1.." + Global::intToString(maxBatchSize)
    );
  }

  // Empty board of the configured size, default rules/params. Outputs are discarded; we only want
  // the forward passes to trigger graph compilation for every batch size that will be seen.
  Board board(nnXLen, nnYLen);
  //Featurize the way this model expects (a no-op under Tromp-Taylorish rules, but robust if the
  //warmup rules ever change).
  BoardHistory history(
    board, P_BLACK, Rules::getTrompTaylorish(), 0,
    BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
  );
  MiscNNInputParams nnInputParams;
  SGFMetadata sgfMeta;
  const SGFMetadata* sgfMetaPtr = NULL;
  if(numInputMetaChannels > 0) {
    sgfMeta = SGFMetadata::makeDummyWarmupProfile();
    sgfMetaPtr = &sgfMeta;
  }

  // Mark the handle as warming up so the backend treats lazy-graph-compilation failures (e.g. cudnn
  // SDPA) leniently, falling back to a custom kernel instead of failing hard. Restored when done.
  bool prevIsWarmup = NeuralNet::setIsWarmup(gpuHandle, true);

  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);

  // Reusable per-row input; identical for every row since it's an empty board.
  std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
  std::vector<NNResultBuf*> resultBufs;
  ownedBufs.reserve(maxBatchSize);
  resultBufs.reserve(maxBatchSize);
  for(int i = 0; i < maxBatchSize; i++) {
    ownedBufs.push_back(std::make_unique<NNResultBuf>());
    NNResultBuf* buf = ownedBufs.back().get();
    fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
    buf->symmetry = 0;
    buf->policyOptimism = nnInputParams.policyOptimism;
    resultBufs.push_back(buf);
  }

  for(int batchSize = 1; batchSize <= maxBatchSize; batchSize++) {
    std::vector<NNOutput*> outputs;
    outputs.reserve(batchSize);
    for(int row = 0; row < batchSize; row++) {
      NNOutput* out = new NNOutput();
      out->nnXLen = nnXLen;
      out->nnYLen = nnYLen;
      out->whiteOwnerMap = NULL;
      outputs.push_back(out);
    }
    NeuralNet::getOutput(gpuHandle, inputBuffers, batchSize, resultBufs.data(), outputs);
    for(NNOutput* out : outputs)
      delete out;
  }

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::setIsWarmup(gpuHandle, prevIsWarmup);
#endif
}

NNEvalBenchmarkResult NNEvaluator::benchmarkPureForward(
  int numWarmups,
  int numIterations,
  const std::vector<int>& boardSizes
) {
  if(numIterations <= 0)
    throw StringError("benchmarkPureForward: numIterations must be positive");
  if(numWarmups < 0)
    throw StringError("benchmarkPureForward: numWarmups must be nonnegative");
  if(boardSizes.size() <= 0)
    throw StringError("benchmarkPureForward: no board sizes specified");
  if(debugSkipNeuralNet || loadedModel == NULL || computeContext == NULL)
    throw StringError("benchmarkPureForward requires a real neural net model");
  if(serverThreads.size() > 0)
    throw StringError("benchmarkPureForward: server threads must not be spawned");
  for(int bSize: boardSizes) {
    if(bSize < 2 || bSize > nnXLen || bSize > nnYLen)
      throw StringError("benchmarkPureForward: board size " + Global::intToString(bSize) + " out of range for NN buffer size");
    if(requireExactNNLen && (bSize != nnXLen || bSize != nnYLen))
      throw StringError("benchmarkPureForward: requireExactNNLen is set but board size " + Global::intToString(bSize) + " does not fill the NN buffer");
  }

  const int numThreadsToUse = (int)gpuIdxByServerThread.size();
  const int batchSize = maxBatchSize;
  testAssert(numThreadsToUse > 0);

  // Left populated after the benchmark returns, deliberately: callers query
  // isAnyThreadUsingFP16() afterward to report what precision was measured.
  {
    std::lock_guard<std::mutex> lock(bufferMutex);
    if(serverThreadsIsUsingFP16.size() < (size_t)numThreadsToUse)
      serverThreadsIsUsingFP16.resize(numThreadsToUse,0);
  }

  NNEvalBenchmarkResult result;
  result.batchSize = batchSize;
  result.numThreads = numThreadsToUse;
  result.numIterations = numIterations;
  result.perThreadIterationSeconds.resize(numThreadsToUse);
  result.perThreadMedianSeconds.assign(numThreadsToUse, 0.0);
  result.perThreadNNEvalsPerSec.assign(numThreadsToUse, 0.0);
  result.sumMedianNNEvalsPerSec = 0.0;
  result.actualWallSeconds = 0.0;
  result.actualWallNNEvalsPerSec = 0.0;

  std::atomic<int> readyCount(0);
  std::atomic<bool> startFlag(false);
  std::atomic<bool> anyError(false);
  std::mutex errorMutex;
  std::exception_ptr firstError;

  // Per-thread end-of-timed-loop timestamps, so that wall time excludes compute handle
  // teardown (freeing the model is slow and an early finisher's frees can also perturb
  // other threads' tail iterations, but that perturbation is at least visible in samples).
  std::vector<std::chrono::steady_clock::time_point> threadEndTimes(numThreadsToUse);

  std::vector<std::thread> threads;
  threads.reserve(numThreadsToUse);
  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++) {
    threads.emplace_back([&,threadIdx]() {
      try {
        ComputeHandle* gpuHandle = NeuralNet::createComputeHandle(
          computeContext,
          loadedModel,
          logger,
          maxBatchSize,
          requireExactNNLen,
          inputsUseNHWC,
          gpuIdxByServerThread[threadIdx],
          threadIdx
        );
        // Declared before the try so that on an exception it is destroyed after the catch frees
        // the compute handle, matching the normal teardown order below.
        std::unique_ptr<NNServerBuf> serverBuf;
        try {
          // Mirror serve(): one compute handle and one set of input buffers per server thread,
          // with the buffers allocated only after createComputeHandle has bound this thread to
          // its GPU (see serve() for why the order matters).
          serverBuf = std::make_unique<NNServerBuf>(*this, loadedModel);
          maybeWarmupComputeHandle(gpuHandle, threadIdx);
          {
            std::lock_guard<std::mutex> lock(bufferMutex);
            if((size_t)threadIdx < serverThreadsIsUsingFP16.size())
              serverThreadsIsUsingFP16[threadIdx] = NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
          }

          MiscNNInputParams nnInputParams;
          SGFMetadata sgfMeta;
          const SGFMetadata* sgfMetaPtr = NULL;
          if(numInputMetaChannels > 0) {
            sgfMeta = SGFMetadata::makeDummyWarmupProfile();
            sgfMetaPtr = &sgfMeta;
          }

          // Empty boards, cycling through the requested sizes across the rows of the batch.
          // Position content doesn't affect speed, but board size does when masking is active.
          std::vector<std::unique_ptr<NNResultBuf>> ownedBufs;
          std::vector<NNResultBuf*> resultBufs;
          ownedBufs.reserve(batchSize);
          resultBufs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            int bSize = boardSizes[row % boardSizes.size()];
            Board board(bSize, bSize);
            BoardHistory history(
              board, P_BLACK, Rules::getTrompTaylorish(), 0,
              BoardHistoryModes(modelPreferPassAliveUnderSuicideRules(), modelPreferExcludeTerritoryAdjacentToAtari())
            );
            ownedBufs.push_back(std::make_unique<NNResultBuf>());
            NNResultBuf* buf = ownedBufs.back().get();
            fillRowBufs(board, history, P_BLACK, sgfMetaPtr, nnInputParams, *buf);
            buf->symmetry = 0;
            buf->policyOptimism = nnInputParams.policyOptimism;
            resultBufs.push_back(buf);
          }

          std::vector<std::unique_ptr<NNOutput>> ownedOutputs;
          std::vector<NNOutput*> outputs;
          ownedOutputs.reserve(batchSize);
          outputs.reserve(batchSize);
          for(int row = 0; row < batchSize; row++) {
            ownedOutputs.push_back(std::make_unique<NNOutput>());
            NNOutput* out = ownedOutputs.back().get();
            out->nnXLen = nnXLen;
            out->nnYLen = nnYLen;
            out->whiteOwnerMap = NULL;
            outputs.push_back(out);
          }

          for(int i = 0; i < numWarmups; i++)
            NeuralNet::getOutput(gpuHandle, serverBuf->inputBuffers, batchSize, resultBufs.data(), outputs);

          readyCount.fetch_add(1);
          while(!startFlag.load(std::memory_order_acquire))
            std::this_thread::yield();

          std::vector<double>& times = result.perThreadIterationSeconds[threadIdx];
          times.reserve(numIterations);
          for(int i = 0; i < numIterations; i++) {
            std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
            NeuralNet::getOutput(gpuHandle, serverBuf->inputBuffers, batchSize, resultBufs.data(), outputs);
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double>(t1-t0).count());
          }
          threadEndTimes[threadIdx] = std::chrono::steady_clock::now();
        }
        catch(...) {
          NeuralNet::freeComputeHandle(gpuHandle);
          throw;
        }
        NeuralNet::freeComputeHandle(gpuHandle);
      }
      catch(...) {
        anyError.store(true);
        std::lock_guard<std::mutex> lock(errorMutex);
        if(firstError == nullptr)
          firstError = std::current_exception();
      }
    });
  }

  while(readyCount.load() < numThreadsToUse && !anyError.load())
    std::this_thread::yield();
  std::chrono::steady_clock::time_point wallStart = std::chrono::steady_clock::now();
  startFlag.store(true, std::memory_order_release);
  for(std::thread& t: threads)
    t.join();

  if(firstError != nullptr)
    std::rethrow_exception(firstError);

  std::chrono::steady_clock::time_point wallEnd = wallStart;
  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++)
    wallEnd = std::max(wallEnd, threadEndTimes[threadIdx]);

  for(int threadIdx = 0; threadIdx < numThreadsToUse; threadIdx++) {
    std::vector<double> sorted = result.perThreadIterationSeconds[threadIdx];
    std::sort(sorted.begin(), sorted.end());
    double median =
      sorted.size() % 2 == 1 ? sorted[sorted.size()/2] :
      0.5 * (sorted[sorted.size()/2-1] + sorted[sorted.size()/2]);
    result.perThreadMedianSeconds[threadIdx] = median;
    result.perThreadNNEvalsPerSec[threadIdx] = median > 0.0 ? (double)batchSize / median : 0.0;
    result.sumMedianNNEvalsPerSec += result.perThreadNNEvalsPerSec[threadIdx];
  }
  result.actualWallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
  if(result.actualWallSeconds > 0.0)
    result.actualWallNNEvalsPerSec =
      (double)numThreadsToUse * (double)batchSize * (double)numIterations / result.actualWallSeconds;
  return result;
}

void NNEvaluator::serve(
  Rand& rand,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  int64_t numBatchesHandledThisThread = 0;
  int64_t numRowsHandledThisThread = 0;

  ComputeHandle* gpuHandle = NULL;
  if(loadedModel != NULL) {
    gpuHandle = NeuralNet::createComputeHandle(
      computeContext,
      loadedModel,
      logger,
      maxBatchSize,
      requireExactNNLen,
      inputsUseNHWC,
      gpuIdxForThisThread,
      serverThreadIdx
    );

    // Warm up lazily-compiled backend graphs before reporting this thread as started.
    maybeWarmupComputeHandle(gpuHandle, serverThreadIdx);
  }

  // Allocate input buffers only after createComputeHandle, which binds this thread to its GPU.
  // On the CUDA backend the pinned host allocation in the buffers initializes a context on the
  // thread's current device, so doing it before the GPU binding would leave a stray context
  // (and some wasted VRAM) on GPU 0 even when only other GPUs are configured.
  NNServerBuf buf(*this, loadedModel);

  {
    lock_guard<std::mutex> lock(bufferMutex);
    testAssert(serverThreadIdx < serverThreadsIsUsingFP16.size());
    serverThreadsIsUsingFP16[serverThreadIdx] = gpuHandle == NULL ? 0 : NeuralNet::isUsingFP16(gpuHandle) ? 1 : 0;
    numServerThreadsStartingUp--;
    if(numServerThreadsStartingUp <= 0)
      mainThreadWaitingForSpawn.notify_all();
  }

  vector<NNResultBuf*> resultBufs;
  resultBufs.reserve(maxBatchSize);

  vector<NNOutput*> outputBuf;

  unique_lock<std::mutex> lock(bufferMutex,std::defer_lock);
  while(true) {
    resultBufs.clear();
    int desiredBatchSize = std::min(maxBatchSize, maxRowsToSendPerBatch.load(std::memory_order_acquire));
    bool gotAnything = queryQueue.waitPopUpToN(resultBufs,desiredBatchSize);
    // Queue being closed is a signal that we're done.
    if(!gotAnything)
      break;

    int numRows = (int)resultBufs.size();
    testAssert(numRows > 0);

    bool doRandomize = currentDoRandomize.load(std::memory_order_acquire);
    int defaultSymmetry = currentDefaultSymmetry.load(std::memory_order_acquire);

    if(debugSkipNeuralNet) {
      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        int boardXSize = resultBuf->boardXSizeForServer;
        int boardYSize = resultBuf->boardYSizeForServer;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::make_shared<NNOutput>();

        float* policyProbs = resultBuf->result->policyProbs;
        for(int i = 0; i<NNPos::MAX_NN_POLICY_SIZE; i++)
          policyProbs[i] = 0;

        // At this point, these aren't probabilities, since this is before the postprocessing
        // that happens for each result. These just need to be unnormalized log probabilities.
        // Illegal move filtering happens later.
        for(int y = 0; y<boardYSize; y++) {
          for(int x = 0; x<boardXSize; x++) {
            int pos = NNPos::xyToPos(x,y,nnXLen);
            policyProbs[pos] = (float)rand.nextGaussian();
          }
        }
        policyProbs[NNPos::locToPos(Board::PASS_LOC,boardXSize,nnXLen,nnYLen)] = (float)rand.nextGaussian();

        resultBuf->result->nnXLen = nnXLen;
        resultBuf->result->nnYLen = nnYLen;
        if(resultBuf->includeOwnerMap) {
          float* whiteOwnerMap = new float[nnXLen*nnYLen];
          for(int i = 0; i<nnXLen*nnYLen; i++)
            whiteOwnerMap[i] = 0.0;
          for(int y = 0; y<boardYSize; y++) {
            for(int x = 0; x<boardXSize; x++) {
              int pos = NNPos::xyToPos(x,y,nnXLen);
              whiteOwnerMap[pos] = (float)rand.nextGaussian() * 0.20f;
            }
          }
          resultBuf->result->whiteOwnerMap = whiteOwnerMap;
        }
        else {
          resultBuf->result->whiteOwnerMap = NULL;
        }

        // These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
        double whiteWinProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteLossProb = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMean = 0.0 + rand.nextGaussian() * 0.20;
        double whiteScoreMeanSq = 0.0 + rand.nextGaussian() * 0.20;
        double whiteNoResultProb = 0.0 + rand.nextGaussian() * 0.20;
        double varTimeLeft = 0.5 * boardXSize * boardYSize;
        resultBuf->result->whiteWinProb = (float)whiteWinProb;
        resultBuf->result->whiteLossProb = (float)whiteLossProb;
        resultBuf->result->whiteNoResultProb = (float)whiteNoResultProb;
        resultBuf->result->whiteScoreMean = (float)whiteScoreMean;
        resultBuf->result->whiteScoreMeanSq = (float)whiteScoreMeanSq;
        resultBuf->result->whiteLead = (float)whiteScoreMean;
        resultBuf->result->varTimeLeft = (float)varTimeLeft;
        resultBuf->result->shorttermWinlossError = 0.0f;
        resultBuf->result->shorttermScoreError = 0.0f;
        resultBuf->result->policyOptimismUsed = (float)resultBuf->policyOptimism;
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }
    else {
      outputBuf.clear();
      for(int row = 0; row<numRows; row++) {
        NNOutput* emptyOutput = new NNOutput();
        testAssert(resultBufs[row] != NULL);
        emptyOutput->nnXLen = nnXLen;
        emptyOutput->nnYLen = nnYLen;
        if(resultBufs[row]->includeOwnerMap)
          emptyOutput->whiteOwnerMap = new float[nnXLen*nnYLen];
        else
          emptyOutput->whiteOwnerMap = NULL;
        outputBuf.push_back(emptyOutput);
      }

      for(int row = 0; row<numRows; row++) {
        if(resultBufs[row]->symmetry == NNInputs::SYMMETRY_NOTSPECIFIED) {
          if(doRandomize)
            resultBufs[row]->symmetry = rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
          else {
            testAssert(defaultSymmetry >= 0 && defaultSymmetry <= SymmetryHelpers::NUM_SYMMETRIES-1);
            resultBufs[row]->symmetry = defaultSymmetry;
          }
        }
      }

      NeuralNet::getOutput(gpuHandle, buf.inputBuffers, numRows, resultBufs.data(), outputBuf);
      testAssert(outputBuf.size() == numRows);

      m_numRowsProcessed.fetch_add(numRows, std::memory_order_relaxed);
      m_numBatchesProcessed.fetch_add(1, std::memory_order_relaxed);
      numRowsHandledThisThread += numRows;
      numBatchesHandledThisThread += 1;

      for(int row = 0; row < numRows; row++) {
        testAssert(resultBufs[row] != NULL);
        NNResultBuf* resultBuf = resultBufs[row];
        resultBufs[row] = NULL;

        unique_lock<std::mutex> resultLock(resultBuf->resultMutex);
        testAssert(resultBuf->hasResult == false);
        resultBuf->result = std::shared_ptr<NNOutput>(outputBuf[row]);
        resultBuf->hasResult = true;
        resultBuf->clientWaitingForResult.notify_all();
        resultLock.unlock();
      }
    }

    // Lock and update stats before looping again
    lock.lock();
    numOngoingEvals -= numRows;

    if(numWaitingEvals > 0) {
      numEvalsToAwaken += numWaitingEvals;
      numWaitingEvals = 0;
      waitingForFinish.notify_all();
    }
    lock.unlock();
    continue;
  }

  NeuralNet::freeComputeHandle(gpuHandle);
  if(logger != NULL) {
    logger->write(
      "GPU " + Global::intToString(gpuIdxForThisThread) + " finishing, processed " +
      Global::int64ToString(numRowsHandledThisThread) + " rows " +
      Global::int64ToString(numBatchesHandledThisThread) + " batches"
    );
  }
}

void NNEvaluator::waitForNextNNEvalIfAny() {
  unique_lock<std::mutex> lock(bufferMutex);
  if(numOngoingEvals <= 0)
    return;

  numWaitingEvals++;
  while(numEvalsToAwaken <= 0 && !isKilled)
    waitingForFinish.wait(lock);
  numEvalsToAwaken--;
}


static double softPlus(double x) {
  // Avoid blowup
  if(x > 40.0)
    return x;
  else
    return log(1.0 + exp(x));
}

static const int daggerPattern[9][8] = {
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,2,1,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,2,1,0,0,0,0,0},
  {0,3,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
};
static bool daggerMatch(const Board& board, Player nextPla, Loc& banned, int symmetry) {
  for(int yi = 0; yi < 9; yi++) {
    for(int xi = 0; xi < 8; xi++) {
      int y = yi;
      int x = xi;
      if((symmetry & 0x1) != 0)
        std::swap(x,y);
      if((symmetry & 0x2) != 0)
        x = board.x_size-1-x;
      if((symmetry & 0x4) != 0)
        y = board.y_size-1-y;
      Loc loc = Location::getLoc(x,y,board.x_size);
      int m = daggerPattern[yi][xi];
      if(m == 0 && board.colors[loc] != C_EMPTY)
        return false;
      if(m == 1 && board.colors[loc] != nextPla)
        return false;
      if(m == 2 && board.colors[loc] != getOpp(nextPla))
        return false;
      if(m == 3)
        banned = loc;
    }
  }
  return true;
}

std::shared_ptr<NNOutput>* NNEvaluator::averageMultipleSymmetries(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& baseNNInputParams,
  NNResultBuf& buf,
  bool includeOwnerMap,
  Rand& rand,
  int numSymmetriesToSample
) {
  MiscNNInputParams nnInputParams = baseNNInputParams;
  vector<std::shared_ptr<NNOutput>> ptrs;
  std::array<int, SymmetryHelpers::NUM_SYMMETRIES> symmetryIndexes;
  std::iota(symmetryIndexes.begin(), symmetryIndexes.end(), 0);
  for(int i = 0; i<numSymmetriesToSample; i++) {
    std::swap(symmetryIndexes[i], symmetryIndexes[rand.nextInt(i,SymmetryHelpers::NUM_SYMMETRIES-1)]);
    nnInputParams.symmetry = symmetryIndexes[i];
    bool skipCacheThisIteration = true; // Skip cache since there's no guarantee which symmetry is in the cache
    evaluate(
      board, history, nextPlayer, sgfMeta,
      nnInputParams,
      buf, skipCacheThisIteration, includeOwnerMap
    );
    ptrs.push_back(std::move(buf.result));
  }
  return new std::shared_ptr<NNOutput>(new NNOutput(ptrs));
}

//-------------------------------------------------------------------------------------
//LZ0 ladder correction v5: Leela Zero-style inputs (V8) have no ladder feature planes, so
//both the policy and the value head are blind to ladders. Instead of hand-tuned penalties:
//  - KataGo's rule-based solver provides exact verdicts (which two-liberty groups are
//    capturable / escapable) and the chase moves;
//  - for a proven-capturable big group the ladder line is played out on the board and the
//    RESOLVED position is evaluated with the SAME net - after the capture the stones are
//    simply gone, so even a ladder-blind net reads the material correctly. That winrate,
//    blended back in, replaces every hand-tuned penalty magnitude (design doc: 05_验证输出/
//    征子修正v3_对齐参考实现设计_20261025.md, v5 section);
//  - policy surgery keeps the reference fork's mild semantics: demote moves that leave a
//    solver-dead group (just below the 3rd-ranked move, probability factor 0.9) and zero
//    futile chases of solver-alive groups, and a boost for escapes the solver proves alive.
//Only used for LZ models; native KataGo nets are unaffected.

namespace LzLadder {
  Params params;
  std::atomic<int> rootArmed(0);
  Params::Params()
    : enabled(true), minStones(1),
      checkNodes(10), minPolicy(5e-4), policyFactor(0.9), penaltyRank(2), chaseStones(4),
      boostFactor(2.0), defenseDepth(11), offenseEnabled(true), offenseDepth(6),
      boostWinrate(0.3), minDepth(6), ladderValueBlend(0.8) {}
}

//Shared certification predicates. The reference fork centralizes "is this a ladder escape"
//in LadderDetection; the boost, the root arming scan and the escaped-group value blend all need
//the same answer, so they call into one place instead of re-deriving it.
namespace LzLadder {
  //First empty liberty of the chain, plus how many empty neighbors the chain has in total.
  static Loc findChainLiberty(const Board& board, Loc loc, int* outNumEmpty) {
    Loc lib = Board::NULL_LOC;
    int numEmpty = 0;
    Loc head = board.chain_head[loc];
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        if(board.chain_head[Location::getLoc(x,y,board.x_size)] != head)
          continue;
        for(int d = 0; d < 4; d++) {
          const int nx = x + (d==0?1:d==1?-1:0);
          const int ny = y + (d==2?1:d==3?-1:0);
          if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= board.y_size)
            continue;
          Loc nb = Location::getLoc(nx,ny,board.x_size);
          if(board.colors[nb] != C_EMPTY)
            continue;
          numEmpty++;
          if(lib == Board::NULL_LOC)
            lib = nb;
        }
      }
    }
    if(outNumEmpty != NULL) *outNumEmpty = numEmpty;
    return lib;
  }

  bool isRealLadderEscape(
    const Board& board, const BoardHistory& history, Loc loc, Player pla,
    int minStones, int minDepth, Loc* outMove, int* outDepth
  ) {
    if(outMove != NULL) *outMove = Board::NULL_LOC;
    if(outDepth != NULL) *outDepth = 0;
    if(loc < 0 || loc >= Board::MAX_ARR_SIZE || board.colors[loc] != pla)
      return false;
    if(board.chain_head[loc] != loc)
      return false;
    if(board.getNumLiberties(loc) != 1 || board.getChainSize(loc) < minStones)
      return false;
    int numEmpty = 0;
    Loc lib = findChainLiberty(board, loc, &numEmpty);
    if(lib == Board::NULL_LOC || numEmpty != 1)
      return false;
    Board trial(board);
    BoardHistory trialHist(history);
    if(!trialHist.makeBoardMoveTolerant(trial,lib,pla))
      return false;
    if(trial.ko_loc != Board::NULL_LOC)
      return false;
    if(trial.getNumLiberties(lib) != 2)
      return false;
    vector<Loc> buf;
    int d = 0;
    if(trial.searchIsLadderCaptured(lib,false,buf,&d))
      return false;
    //Depth 0 means the solver ran out of its fixed node budget (upstream's 25000): the verdict is
    //unknown. The depth also tells how long the chase is; only long ladders need the value
    //correction, short ones are visible to the net and to a shallow search.
    if(d < minDepth)
      return false;
    if(outMove != NULL) *outMove = lib;
    if(outDepth != NULL) *outDepth = d;
    return true;
  }

  bool isEscapedTwoLibertyGroup(
    const Board& board, Loc loc, Player pla, int minStones, int minDepth,
    Loc* outMove, int* outDepth
  ) {
    if(outMove != NULL) *outMove = Board::NULL_LOC;
    if(outDepth != NULL) *outDepth = 0;
    if(loc < 0 || loc >= Board::MAX_ARR_SIZE || board.colors[loc] != pla)
      return false;
    if(board.chain_head[loc] != loc)
      return false;
    if(board.getNumLiberties(loc) != 2 || board.getChainSize(loc) < minStones)
      return false;
    vector<Loc> buf;
    int d = 0;
    Board probe(board);
    if(probe.searchIsLadderCaptured(loc,false,buf,&d))
      return false;
    if(d < minDepth)
      return false;
    if(outMove != NULL) *outMove = findChainLiberty(board, loc, NULL);
    if(outDepth != NULL) *outDepth = d;
    return true;
  }
}

static void applyLZLadderCorrectionV5(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  int nnXLen,
  int nnYLen,
  float* policy,
  const bool* isLegal,
  float* winMulOut
) {
  const LzLadder::Params& P = LzLadder::params;
  const int policySize = NNPos::getPolicySize(nnXLen,nnYLen);

  float maxLogit = -1e25f;
  for(int pos = 0; pos<policySize; pos++) {
    if(!isLegal[pos])
      continue;
    if(policy[pos] > maxLogit)
      maxLogit = policy[pos];
  }
  if(maxLogit <= -1e24f)
    return;

  const bool suicideLegal = history.rules.multiStoneSuicideLegal;
  static const int NDX[4] = {1,-1,0,0};
  static const int NDY[4] = {0,0,1,-1};
  vector<Loc> buf;
  vector<Loc> workingMoves;

  //Reference-fork (LK/lzl-src LadderDetection) semantics, ported:
  //  boost  : our move puts an OPPONENT group into atari (1 liberty, nothing captured) and that
  //           group cannot run the ladder -> winning ladder capture, lift this move's prior.
  //  demote : our move leaves an OWN two-liberty group that the opponent can ladder-capture and
  //           the losing line is deep enough and materially bad enough to be worth pricing.
  //  chase  : "do not chase a group that escapes" stays DISABLED by default - we have no
  //           chase-depth measure, and without it the rule over-triggers and removes good ataris.
  //---------------------------------------------------------------------------
  //Group-driven rescue boost. The candidate-driven pass below only examines the top
  //`checkNodes` priors, so a correct escape move whose prior sits 30 ranks down is never
  //looked at (measured: the escape in the yin-zheng position has a 0.37% prior). Walk the
  //board's own one-liberty groups instead and promote the single move that saves them:
  //our group is in atari, the extension is solver-certified as un-capturable, and the group
  //reaches exactly two liberties (a real ladder escape). The "three or more liberties" case is
  //deliberately excluded - it fires on every dead stone that merely merges into a bigger
  //friendly group.
  if((float)P.boostFactor > 1.0f) {
    const auto boostAt = [&](Loc loc) {
      const int p = NNPos::locToPos(loc, board.x_size, nnXLen, nnYLen);
      if(p < 0 || p >= policySize || !isLegal[p])
        return;
      const float boostLogit = maxLogit + std::log((float)P.boostFactor);
      if(boostLogit > policy[p])
        policy[p] = boostLogit;
    };
    //Rescue boost: our one-liberty group's saving extension. Same long-ladder depth floor as the
    //root gate and the value blend - a short escape is visible to the value head, so lifting its
    //prior would only bias the search.
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        Loc move = Board::NULL_LOC;
        if(LzLadder::isRealLadderEscape(board, history, loc, nextPlayer, P.minStones, P.minDepth, &move, NULL))
          boostAt(move);
      }
    }
  }

  vector<pair<float,int>> candidates;
  candidates.reserve(policySize);
  for(int pos = 0; pos<policySize; pos++) {
    if(!isLegal[pos])
      continue;
    if(NNPos::isPassPos(pos,nnXLen,nnYLen))
      continue;
    candidates.push_back(make_pair(policy[pos],pos));
  }
  if(candidates.empty())
    return;

  float refLogit = candidates[std::min((size_t)P.penaltyRank, candidates.size()-1)].first;
  const float demoteLogit = refLogit + std::log((float)P.policyFactor);

  int numCandidates = std::min((int)candidates.size(), P.checkNodes);
  std::partial_sort(
    candidates.begin(), candidates.begin() + numCandidates, candidates.end(),
    [](const pair<float,int>& a, const pair<float,int>& b) { return a.first > b.first; }
  );

  for(int i = 0; i<numCandidates; i++) {
    const float logit = candidates[i].first;
    if(logit < maxLogit + std::log((float)P.minPolicy))
      break;
    const int pos = candidates[i].second;
    Loc moveLoc = NNPos::posToLoc(pos,board.x_size,board.y_size,nnXLen,nnYLen);
    if(moveLoc == Board::NULL_LOC || moveLoc == Board::PASS_LOC)
      continue;

    const int x = Location::getX(moveLoc,board.x_size);
    const int y = Location::getY(moveLoc,board.x_size);
    //Cheap precheck: only moves that touch a group with at most two liberties can change a
    //ladder verdict, so skip the copy and the solver for everything else.
    bool nearAtari = false;
    for(int d = 0; d<4; d++) {
      const int nx = x + NDX[d];
      const int ny = y + NDY[d];
      if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= board.y_size)
        continue;
      Loc nb = Location::getLoc(nx,ny,board.x_size);
      //<=3, not <=2: our move can drop the group by one liberty (or more if it also captures),
      //so a group that ends up in atari may well have had three liberties before the move.
      if(board.colors[nb] != C_EMPTY && board.getNumLiberties(nb) <= 3) {
        nearAtari = true;
        break;
      }
    }
    if(!nearAtari)
      continue;

    Board copy(board);
    BoardHistory copyHist(history);
    if(!copyHist.makeBoardMoveTolerant(copy,moveLoc,nextPlayer))
      continue;
    //Kos make the ladder verdict unreliable (the recapture might be illegal), so skip them.
    if(copy.ko_loc != Board::NULL_LOC)
      continue;

    const int captured = nextPlayer == P_WHITE
      ? (int)(copy.numWhiteCaptures - board.numWhiteCaptures)
      : (int)(copy.numBlackCaptures - board.numBlackCaptures);

    Loc checkedHeads[4];
    int numChecked = 0;
    bool doBoost = false;
    bool doDemote = false;
    for(int d = 0; d<4; d++) {
      const int nx = x + NDX[d];
      const int ny = y + NDY[d];
      if(nx < 0 || nx >= board.x_size || ny < 0 || ny >= board.y_size)
        continue;
      Loc nb = Location::getLoc(nx,ny,board.x_size);
      Color col = copy.colors[nb];
      if(col == C_EMPTY)
        continue;
      Loc head = copy.chain_head[nb];
      bool seen = false;
      for(int j = 0; j<numChecked; j++) {
        if(checkedHeads[j] == head) { seen = true; break; }
      }
      if(seen)
        continue;
      if(numChecked < 4)
        checkedHeads[numChecked++] = head;

      const int postLibs = copy.getNumLiberties(nb);
      const int preLibs = board.getNumLiberties(nb);
      if(postLibs > 2)
        continue;

      if(col != nextPlayer) {
        const int chainSize = copy.getChainSize(nb);
        //"Complete the ladder" from the defender's turn: he extends at his only liberty, then the
        //chase is played out. Gives BOTH numbers at once - captured (and how many plies to the
        //capture) or escaped (and how many plies the defender ran).
        //The solver itself now reports the depth of its verdict (reference LadderDetection's
        //max_depth / dead_depth): d < 0 = captured in -d plies, d > 0 = escaped, d = 0 unknown.
        int d = 0;
        bool capturedLine = false;
        if(captured == 0 && postLibs <= 1)
          capturedLine = copy.searchIsLadderCaptured(nb,true,buf,&d);
        //Reference boost: our move puts an opponent group into atari that cannot escape. Only a
        //LONG ladder capture needs the prior lift: a short capture is visible to the value head.
        //The solver's depth is negative here (plies to the capture). No group-size restriction:
        //a single stone in a long ladder is exactly the case the value head cannot see.
        if(captured == 0 && postLibs == 1 && capturedLine && d <= -P.minDepth)
          doBoost = true;
        //Reference offence penalty: our atari hits a group that gets out, after a long chase.
        if(P.offenseEnabled && captured == 0 && postLibs == 1
           && chainSize >= P.chaseStones && !capturedLine && d >= P.offenseDepth)
          doDemote = true;
      }
      else {
        //Reference defence rule: our move leaves an own two-liberty group that the opponent
        //captures (depth solver, defender to move first), the losing line is deep enough, and it
        //costs more stones than the captures we just made.
        const int chainSize = copy.getChainSize(nb);
        if(postLibs == 2 && chainSize > captured && chainSize >= P.minStones) {
          //Our group is at two liberties and it is the attacker's turn: the solver's
          //attacker-first form is the right gate, then "complete the ladder" gives the ply count
          //to the capture (that is the same number the reference's IsLadderEscape reports negated).
          //Attacker to move on our two-liberty group: same solver, depth included.
          int d = 0;
          if(copy.searchIsLadderCaptured(nb,false,buf,&d)) {
            const int plies = -d;
            if(plies >= P.defenseDepth && (plies/2 + chainSize) > captured)
              doDemote = true;
          }
        }
      }
    }

    if(doBoost && (float)P.boostFactor > 1.0f) {
      const float boostLogit = maxLogit + std::log((float)P.boostFactor);
      if(boostLogit > policy[pos]) {
        //Reference winrate surgery: the winrate change is scaled by the move's own probability
        //(a low-prior move gets a small nudge, a high-prior one a bigger one), so the position
        //value stays honest. Original probability is logit-space softmax over legal moves.
        if(winMulOut != NULL && P.boostWinrate > 0.0f) {
          float sum = 0.0f;
          for(int q = 0; q < policySize; q++) {
            if(isLegal[q])
              sum += std::exp(policy[q] - maxLogit);
          }
          if(sum > 0.0f) {
            const float p = std::exp(policy[pos] - maxLogit) / sum;
            *winMulOut *= (1.0f + p * (float)P.boostWinrate);
          }
        }
        policy[pos] = boostLogit;
      }
    }
    else if(doDemote) {
      //Reference penalty semantics: cfg_ladder_penalty_base > 0 demotes the move to just below
      //the penaltyRank-th candidate times policyFactor; only penalty_base == 0 zeroes it outright.
      if(P.penaltyRank > 0) {
        if(demoteLogit < policy[pos])
          policy[pos] = demoteLogit;
      }
      else {
        policy[pos] = -1e30f;
      }
    }
  }
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  evaluate(
    board,
    history,
    nextPlayer,
    NULL,
    nnInputParams,
    buf,
    skipCache,
    includeOwnerMap
  );
}

void NNEvaluator::evaluate(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParamsArg,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  testAssert(!isKilled);
  buf.hasResult = false;
  float ladderWinMul = 1.0f;   //multiplicative winrate nudge from the ladder boost (1.0 = none)

  if(board.x_size > nnXLen || board.y_size > nnYLen)
    throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                      " nnYLen = " + Global::intToString(nnYLen) +
                      " but was asked to evaluate board with larger x or y size");
  if(requireExactNNLen) {
    if(board.x_size != nnXLen || board.y_size != nnYLen)
      throw StringError("NNEvaluator was configured with nnXLen = " + Global::intToString(nnXLen) +
                        " nnYLen = " + Global::intToString(nnYLen) +
                        " and requireExactNNLen, but was asked to evaluate board with different x or y size");
  }

  // Avoid using policy optimism for humanSL
  MiscNNInputParams nnInputParams = nnInputParamsArg;
  if(numInputMetaChannels > 0)
    nnInputParams.policyOptimism = 0.0;

  Hash128 nnHash = NNInputs::getHash(board, history, nextPlayer, nnInputParams);
  if(numInputMetaChannels > 0) {
    if(sgfMeta == NULL)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not provided");
    if(!sgfMeta->initialized)
      Global::fatalError("SGFMetadata is required for " + modelName + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?");
    nnHash ^= sgfMeta->getHash(nextPlayer);
  }

  bool hadResultWithoutOwnerMap = false;
  shared_ptr<NNOutput> resultWithoutOwnerMap;
  if(nnCacheTable != NULL && !skipCache && nnCacheTable->get(nnHash,buf.result)) {
    if(!(includeOwnerMap && buf.result->whiteOwnerMap == NULL))
    {
      m_numCacheHits.fetch_add(1, std::memory_order_relaxed);
      buf.hasResult = true;
      return;
    }
    else {
      hadResultWithoutOwnerMap = true;
      resultWithoutOwnerMap = std::move(buf.result);
      buf.result = nullptr;
    }
  }
  buf.includeOwnerMap = includeOwnerMap;

  buf.boardXSizeForServer = board.x_size;
  buf.boardYSizeForServer = board.y_size;

  if(!debugSkipNeuralNet) {
    fillRowBufs(board, history, nextPlayer, sgfMeta, nnInputParams, buf);
  }

  buf.symmetry = nnInputParams.symmetry;
  buf.policyOptimism = nnInputParams.policyOptimism;

  unique_lock<std::mutex> lock(bufferMutex);
  numOngoingEvals += 1;
  lock.unlock();

  bool suc = queryQueue.forcePush(&buf);
  testAssert(suc);

  unique_lock<std::mutex> resultLock(buf.resultMutex);
  while(!buf.hasResult)
    buf.clientWaitingForResult.wait(resultLock);
  resultLock.unlock();

  // Perform postprocessing on the result - turn the nn output into probabilities
  // As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
  // and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
  // and causing policy weights to be different, which would reduce performance of successive searches in a game
  // by making the successive searches distribute their playouts less coherently and using the cache more poorly.
  if(hadResultWithoutOwnerMap) {
    buf.result->whiteWinProb = resultWithoutOwnerMap->whiteWinProb;
    buf.result->whiteLossProb = resultWithoutOwnerMap->whiteLossProb;
    buf.result->whiteNoResultProb = resultWithoutOwnerMap->whiteNoResultProb;
    buf.result->whiteScoreMean = resultWithoutOwnerMap->whiteScoreMean;
    buf.result->whiteScoreMeanSq = resultWithoutOwnerMap->whiteScoreMeanSq;
    buf.result->whiteLead = resultWithoutOwnerMap->whiteLead;
    buf.result->varTimeLeft = resultWithoutOwnerMap->varTimeLeft;
    buf.result->shorttermWinlossError = resultWithoutOwnerMap->shorttermWinlossError;
    buf.result->shorttermScoreError = resultWithoutOwnerMap->shorttermScoreError;
    std::copy(resultWithoutOwnerMap->policyProbs, resultWithoutOwnerMap->policyProbs + NNPos::MAX_NN_POLICY_SIZE, buf.result->policyProbs);
    buf.result->policyOptimismUsed = (float)resultWithoutOwnerMap->policyOptimismUsed;
    buf.result->nnXLen = resultWithoutOwnerMap->nnXLen;
    buf.result->nnYLen = resultWithoutOwnerMap->nnYLen;
    testAssert(buf.result->whiteOwnerMap != NULL);
  }
  else {
    float* policy = buf.result->policyProbs;

    float policyOutputScaling = postProcessParams.outputScaleMultiplier / nnInputParams.nnPolicyTemperature;

    int xSize = board.x_size;
    int ySize = board.y_size;

    float maxPolicy = -1e25f;
    bool isLegal[NNPos::MAX_NN_POLICY_SIZE];
    int legalCount = 0;
    testAssert(nextPlayer == history.presumedNextMovePla);
    for(int i = 0; i<policySize; i++) {
      Loc loc = NNPos::posToLoc(i,xSize,ySize,nnXLen,nnYLen);
      isLegal[i] = history.isLegal(board,loc,nextPlayer);
    }

    if(nnInputParams.avoidMYTDaggerHack && xSize >= 13 && ySize >= 13) {
      for(int symmetry = 0; symmetry < 8; symmetry++) {
        Loc banned = Board::NULL_LOC;
        if(daggerMatch(board, nextPlayer, banned, symmetry)) {
          if(banned != Board::NULL_LOC) {
            isLegal[NNPos::locToPos(banned,xSize,nnXLen,nnYLen)] = false;
          }
        }
      }
    }

    for(int i = 0; i<policySize; i++) {
      float policyValue;
      if(isLegal[i]) {
        legalCount += 1;
        policyValue = policy[i] * policyOutputScaling;
      }
      else
        policyValue = -1e30f;

      policy[i] = policyValue;
      if(policyValue > maxPolicy)
        maxPolicy = policyValue;
    }

    testAssert(legalCount > 0);

    //LZ0 ladder correction v5, policy part: solver-verdict demotions on the scaled logits
    //(before the softmax, legality mask in place). The value override happens one level up in
    //evaluateWithLzLadder. Native KataGo nets unaffected.
    //LZ_NO_LADDER=1 skips it entirely (A/B experiment switch for strength testing).
    if(isLZModelFlag && LzLadder::params.enabled && getenv("LZ_NO_LADDER") == NULL)
      applyLZLadderCorrectionV5(board,history,nextPlayer,nnXLen,nnYLen,policy,isLegal,&ladderWinMul);

    //The correction can change which logit is the largest, so recompute the softmax reference.
    maxPolicy = -1e25f;
    for(int i = 0; i<policySize; i++) {
      if(policy[i] > maxPolicy)
        maxPolicy = policy[i];
    }

    float policySum = 0.0f;

    if(nnInputParams.enablePassingHacks) {
      //Cap passing prior policy at 95% (19x other moves)
      float maxPassPolicySumFactor = 19.0f;

      for(int i = 0; i<policySize-1; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
      int passPos = NNPos::locToPos(Board::PASS_LOC, xSize, nnXLen, nnYLen);
      testAssert(passPos == policySize-1);
      int i = passPos;
      policy[i] = std::max(1e-20f, std::min(exp(policy[i] - maxPolicy), policySum * maxPassPolicySumFactor));
      policySum += policy[i];
    }
    else {
      for(int i = 0; i<policySize; i++) {
        policy[i] = exp(policy[i] - maxPolicy);
        policySum += policy[i];
      }
    }

    if(!isfinite(policySum)) {
      cout << "Got nonfinite for policy sum" << endl;
      history.printDebugInfo(cout,board);
      throw StringError("Got nonfinite for policy sum");
    }

    // Somehow all legal moves rounded to 0 probability
    if(policySum <= 0.0) {
      if(!buf.errorLogLockout && logger != NULL) {
        buf.errorLogLockout = true;
        logger->write("Warning: all legal moves rounded to 0 probability for " + string(modelFileName));
      }
      float uniform = 1.0f / legalCount;
      for(int i = 0; i<policySize; i++) {
        policy[i] = isLegal[i] ? uniform : -1.0f;
      }
    }
    // Normal case
    else {
      for(int i = 0; i<policySize; i++)
        policy[i] = isLegal[i] ? (policy[i] / policySum) : -1.0f;
    }

    // Fill everything out-of-bounds too, for robustness.
    for(int i = policySize; i<NNPos::MAX_NN_POLICY_SIZE; i++)
      policy[i] = -1.0f;

    buf.result->policyOptimismUsed = (float)nnInputParams.policyOptimism;

    // Fix up the value as well. Note that the neural net gives us back the value from the perspective
    // of the player so we need to negate that to make it the white value.
    if(modelVersion == 3) {
      const double twoOverPi = 0.63661977236758134308;

      double winProb;
      double lossProb;
      double noResultProb;
      // Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
      double scoreValue = atan(buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier) * twoOverPi;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;

        // Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        if(!isfinite(probSum) || !isfinite(scoreValue)) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits << " " << scoreValue << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      if(nextPlayer == P_WHITE) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)ScoreValue::approxWhiteScoreOfScoreValueSmooth(scoreValue,0.0,2.0,board.sqrtBoardArea());
        buf.result->whiteScoreMeanSq = buf.result->whiteScoreMean * buf.result->whiteScoreMean;
        buf.result->whiteLead = buf.result->whiteScoreMean;
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }

    }
    else if(modelVersion >= 4) {
      double winProb;
      double lossProb;
      double noResultProb;
      double scoreMean;
      double scoreMeanSq;
      double lead;
      double varTimeLeft;
      double shorttermWinlossError;
      double shorttermScoreError;
      {
        double winLogits = buf.result->whiteWinProb * postProcessParams.outputScaleMultiplier;
        double lossLogits = buf.result->whiteLossProb * postProcessParams.outputScaleMultiplier;
        double noResultLogits = buf.result->whiteNoResultProb * postProcessParams.outputScaleMultiplier;
        double scoreMeanPreScaled = buf.result->whiteScoreMean * postProcessParams.outputScaleMultiplier;
        double scoreStdevPreSoftplus = buf.result->whiteScoreMeanSq * postProcessParams.outputScaleMultiplier;
        double leadPreScaled = buf.result->whiteLead * postProcessParams.outputScaleMultiplier;
        double varTimeLeftPreSoftplus = buf.result->varTimeLeft * postProcessParams.outputScaleMultiplier;
        double shorttermWinlossErrorPreSoftplus = buf.result->shorttermWinlossError * postProcessParams.outputScaleMultiplier;
        double shorttermScoreErrorPreSoftplus = buf.result->shorttermScoreError * postProcessParams.outputScaleMultiplier;

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultLogits -= 100000.0;

        // Softmax
        double maxLogits = std::max(std::max(winLogits,lossLogits),noResultLogits);
        winProb = exp(winLogits - maxLogits);
        lossProb = exp(lossLogits - maxLogits);
        noResultProb = exp(noResultLogits - maxLogits);

        if(history.rules.koRule != Rules::KO_SIMPLE && history.rules.scoringRule != Rules::SCORING_TERRITORY)
          noResultProb = 0.0;

        double probSum = winProb + lossProb + noResultProb;
        winProb /= probSum;
        lossProb /= probSum;
        noResultProb /= probSum;

        scoreMean = scoreMeanPreScaled * postProcessParams.scoreMeanMultiplier;
        double scoreStdev = softPlus(scoreStdevPreSoftplus) * postProcessParams.scoreStdevMultiplier;
        scoreMeanSq = scoreMean * scoreMean + scoreStdev * scoreStdev;
        lead = leadPreScaled * postProcessParams.leadMultiplier;
        varTimeLeft = softPlus(varTimeLeftPreSoftplus) * postProcessParams.varianceTimeMultiplier;

        // scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
        // noResult counts as 0 score for scorevalue purposes.
        scoreMean = scoreMean * (1.0-noResultProb);
        scoreMeanSq = scoreMeanSq * (1.0-noResultProb);
        lead = lead * (1.0-noResultProb);

        if(modelVersion >= 14) {
          {
            double s = softPlus(shorttermWinlossErrorPreSoftplus * 0.5);
            shorttermWinlossError = sqrt(s * s * postProcessParams.shorttermValueErrorMultiplier);
          }
          {
            double s = softPlus(shorttermScoreErrorPreSoftplus * 0.5);
            shorttermScoreError = sqrt(s * s * postProcessParams.shorttermScoreErrorMultiplier);
          }
        }
        else if(modelVersion >= 10) {
          shorttermWinlossError = sqrt(softPlus(shorttermWinlossErrorPreSoftplus) * postProcessParams.shorttermValueErrorMultiplier);
          shorttermScoreError = sqrt(softPlus(shorttermScoreErrorPreSoftplus) * postProcessParams.shorttermScoreErrorMultiplier);
        }
        else {
          shorttermWinlossError = softPlus(shorttermWinlossErrorPreSoftplus);
          shorttermScoreError = softPlus(shorttermScoreErrorPreSoftplus) * 10.0;
        }

        if(
          !isfinite(probSum) ||
          !isfinite(scoreMean) ||
          !isfinite(scoreMeanSq) ||
          !isfinite(lead) ||
          !isfinite(varTimeLeft) ||
          !isfinite(shorttermWinlossError) ||
          !isfinite(shorttermScoreError)
        ) {
          cout << "Got nonfinite for nneval value" << endl;
          cout << winLogits << " " << lossLogits << " " << noResultLogits
               << " " << scoreMean << " " << scoreMeanSq
               << " " << lead << " " << varTimeLeft
               << " " << shorttermWinlossError << " " << shorttermScoreError
               << endl;
          throw StringError("Got nonfinite for nneval value");
        }
      }

      //Leela Zero-style networks have no score, variance, or shortterm error outputs (their
      //score head is a dummy). Neutralize those fields: the score is unknown, so report a draw
      //with zero variance, and mark the uncertainty fields as unsupported (-1).
      if(isLZModelFlag) {
        scoreMean = 0.0;
        scoreMeanSq = 0.0;
        lead = 0.0;
        varTimeLeft = -1.0;
        shorttermWinlossError = -1.0;
        shorttermScoreError = -1.0;
      }

      //The neural net's (win,loss,draw) triple is from the perspective of the player to move by
      //default. Model version 18 nets may instead report it from black's perspective
      //(valueFromBlackPerspective, e.g. Leela Zero v2/ELF nets, whose raw output is the
      //black win probability regardless of whose turn it is).
      //  - player-to-move perspective: whiteWinProb = P(white wins) = (nextPlayer == P_WHITE)
      //    ? winProb : lossProb, i.e. swap exactly when black is to move.
      //  - black perspective: whiteWinProb = P(white wins) = lossProb unconditionally, i.e.
      //    always swap.
      bool swapWinLoss = valueFromBlackPerspective ? true : (nextPlayer != P_WHITE);
      if(!swapWinLoss) {
        buf.result->whiteWinProb = (float)winProb;
        buf.result->whiteLossProb = (float)lossProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = (float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = (float)lead;
      }
      else {
        buf.result->whiteWinProb = (float)lossProb;
        buf.result->whiteLossProb = (float)winProb;
        buf.result->whiteNoResultProb = (float)noResultProb;
        buf.result->whiteScoreMean = -(float)scoreMean;
        buf.result->whiteScoreMeanSq = (float)scoreMeanSq;
        buf.result->whiteLead = -(float)lead;
      }

      if(modelVersion >= 9) {
        buf.result->varTimeLeft = (float)varTimeLeft;
        buf.result->shorttermWinlossError = (float)shorttermWinlossError;
        buf.result->shorttermScoreError = (float)shorttermScoreError;
      }
      else {
        buf.result->varTimeLeft = -1;
        buf.result->shorttermWinlossError = -1;
        buf.result->shorttermScoreError = -1;
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  // Postprocess ownermap
  if(buf.result->whiteOwnerMap != NULL) {
    if(modelVersion >= 3) {
      for(int pos = 0; pos<nnXLen*nnYLen; pos++) {
        int y = pos / nnXLen;
        int x = pos % nnXLen;
        if(y >= board.y_size || x >= board.x_size)
          buf.result->whiteOwnerMap[pos] = 0.0f;
        else {
          // Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
          // but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
          if(nextPlayer == P_WHITE)
            buf.result->whiteOwnerMap[pos] = tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
          else
            buf.result->whiteOwnerMap[pos] = -tanh(buf.result->whiteOwnerMap[pos] * postProcessParams.outputScaleMultiplier);
        }
      }
    }
    else {
      throw StringError("NNEval value postprocessing not implemented for model version");
    }
  }

  //Ladder boost's winrate surgery (see applyLZLadderCorrectionV5): multiplicative, in
  //side-to-move terms, so it commutes with the white/black perspective of whiteWinProb.
  if(ladderWinMul != 1.0f) {
    float stmWin = (nextPlayer == P_WHITE) ? buf.result->whiteWinProb : (1.0f - buf.result->whiteWinProb);
    stmWin = std::min(1.0f, std::max(0.0f, stmWin * ladderWinMul));
    buf.result->whiteWinProb = (nextPlayer == P_WHITE) ? stmWin : (1.0f - stmWin);
    buf.result->whiteLossProb = 1.0f - buf.result->whiteWinProb - buf.result->whiteNoResultProb;
  }

  // And record the nnHash in the result and put it into the table
  buf.result->nnHash = nnHash;
  if(nnCacheTable != NULL)
    nnCacheTable->set(buf.result);

}

//v5 LZ ladder value override, search-facing wrapper: after the ordinary evaluate() returns,
//look for a solver-proven capturable two-liberty group of the opponent's with ourselves to
//move (attacker first). If one exists, play the ladder out on the board and evaluate the
//RESOLVED position with the same net as a clean sequential call (evaluating from inside the
//postprocessing of evaluate() would deadlock), then blend that winrate into this result.
//After the capture the stones are simply off the board, so even a ladder-blind value head
//reads the material - the magnitude always comes from the net itself, never a constant.
void NNEvaluator::evaluateWithLzLadder(
  const Board& board,
  const BoardHistory& history,
  Player nextPlayer,
  const SGFMetadata* sgfMeta,
  const MiscNNInputParams& nnInputParams,
  NNResultBuf& buf,
  bool skipCache,
  bool includeOwnerMap
) {
  evaluate(board, history, nextPlayer, sgfMeta, nnInputParams, buf, skipCache, includeOwnerMap);

  //Value half of the ladder correction, gated to the exact state where it matters: the opponent
  //(the side that is NOT to move) has a group with exactly two liberties and the depth solver -
  //with US moving first - says it cannot be run down. That is precisely "their escape already
  //happened": policy alone is not enough there, because the boosted move gets searched first but
  //the ladder-blind value head still calls two liberties "dead" and the search drops it again
  //(measured on the yin-zheng position: escape prior 0.37% -> 30.5%, value 52.3% vs 10b's 77.0%).
  //The correction prices the escaped board: give the defender the extension it is entitled to and
  //evaluate that, so the node value reflects a live group. Only LONG ladders qualify (remaining
  //escape depth >= lzLadderMinDepth): a short escape reaches three liberties within a move or two
  //and the value head already sees it.
  if(!isLZModelFlag || !LzLadder::params.enabled || getenv("LZ_NO_LADDER") != NULL)
    return;
  if(buf.result == NULL)
    return;
  if(board.ko_loc != Board::NULL_LOC)
    return;
  const LzLadder::Params& P = LzLadder::params;
  if((float)P.ladderValueBlend <= 0.0f)
    return;
  //Only while the current search's ROOT contains a ladder shape (see rootArmed). This keeps every
  //ordinary search free of the extra solver calls.
  if(LzLadder::rootArmed.load(std::memory_order_relaxed) == 0)
    return;
  {
    const Player opp = getOpp(nextPlayer);
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        Loc lib = Board::NULL_LOC;
        int d = 0;
        //Same "real long-ladder escape" predicate as the boost and the root arming scan: the
        //attacker may still be unable to run it down, and at least lzLadderMinDepth plies of
        //chase must remain. Near the end of a long ladder the remaining depth falls below that,
        //but then both sides only have forced moves and a shallow search sees the capture.
        if(!LzLadder::isEscapedTwoLibertyGroup(board, loc, opp, P.minStones, P.minDepth, &lib, &d))
          continue;
        if(lib == Board::NULL_LOC)
          continue;
        const int chainSize = board.getChainSize(loc);
        Board trial(board);
        BoardHistory trialHist(history);
        if(!trialHist.makeBoardMoveTolerant(trial,lib,opp))
          continue;
        MiscNNInputParams endParams = nnInputParams;
        endParams.symmetry = 0;
        NNResultBuf endBuf;
        try {
          evaluate(trial, trialHist, nextPlayer, sgfMeta, endParams, endBuf, false, false);
          const float dEnd = endBuf.result->whiteWinProb;
          const float blend = (float)P.ladderValueBlend;
          const float blended = (1.0f - blend) * buf.result->whiteWinProb + blend * dEnd;
          if(getenv("LZ_LADDER_DEBUG") != NULL)
            std::fprintf(stderr, "[LADVE] escaped-group d=%d size=%d raw=%.4f end=%.4f -> %.4f\n",
              d, chainSize, buf.result->whiteWinProb, dEnd, blended);
          buf.result->whiteWinProb = blended;
          buf.result->whiteLossProb = 1.0f - blended - buf.result->whiteNoResultProb;
        }
        catch(const std::exception&) {
        }
        return;
      }
    }
  }
}
// Uncomment this to lower the effective hash size down to one where we get true collisions
// #define SIMULATE_TRUE_HASH_COLLISIONS

NNCacheTable::Entry::Entry()
  :ptr(nullptr)
{}
NNCacheTable::Entry::~Entry()
{}

NNCacheTable::NNCacheTable(int sizePowerOfTwo, int mutexPoolSizePowerOfTwo) {
  if(sizePowerOfTwo < 0 || sizePowerOfTwo > 63)
    throw StringError("NNCacheTable: Invalid sizePowerOfTwo: " + Global::intToString(sizePowerOfTwo));
  if(mutexPoolSizePowerOfTwo < 0 || mutexPoolSizePowerOfTwo > 31)
    throw StringError("NNCacheTable: Invalid mutexPoolSizePowerOfTwo: " + Global::intToString(mutexPoolSizePowerOfTwo));
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  sizePowerOfTwo = sizePowerOfTwo > 12 ? 12 : sizePowerOfTwo;
#endif
  if(mutexPoolSizePowerOfTwo > sizePowerOfTwo)
    mutexPoolSizePowerOfTwo = sizePowerOfTwo;

  tableSize = ((uint64_t)1) << sizePowerOfTwo;
  tableMask = tableSize-1;
  entries = new Entry[tableSize];
  uint32_t mutexPoolSize = ((uint32_t)1) << mutexPoolSizePowerOfTwo;
  mutexPoolMask = mutexPoolSize-1;
  mutexPool = new MutexPool(mutexPoolSize);
}
NNCacheTable::~NNCacheTable() {
  delete[] entries;
  delete mutexPool;
}

bool NNCacheTable::get(Hash128 nnHash, shared_ptr<NNOutput>& ret) {
  // Free ret BEFORE locking, to avoid any expensive operations while locked.
  if(ret != nullptr)
    ret.reset();

  uint64_t idx = nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  std::lock_guard<std::mutex> lock(mutex);

  bool found = false;
#if defined(SIMULATE_TRUE_HASH_COLLISIONS)
  if(entry.ptr != nullptr && ((entry.ptr->nnHash.hash0 ^ nnHash.hash0) & 0xFFF) == 0) {
    ret = entry.ptr;
    found = true;
  }
#else
  if(entry.ptr != nullptr && entry.ptr->nnHash == nnHash) {
    ret = entry.ptr;
    found = true;
  }
#endif
  return found;
}

void NNCacheTable::set(const shared_ptr<NNOutput>& p) {
  // Immediately copy p right now, before locking, to avoid any expensive operations while locked.
  shared_ptr<NNOutput> buf(p);

  uint64_t idx = p->nnHash.hash0 & tableMask;
  uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
  Entry& entry = entries[idx];
  std::mutex& mutex = mutexPool->getMutex(mutexIdx);

  {
    std::lock_guard<std::mutex> lock(mutex);
    // Perform a swap, to avoid any expensive free under the mutex.
    entry.ptr.swap(buf);
  }

  // No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
}

void NNCacheTable::clear() {
  shared_ptr<NNOutput> buf;
  for(size_t idx = 0; idx<tableSize; idx++) {
    Entry& entry = entries[idx];
    uint32_t mutexIdx = (uint32_t)idx & mutexPoolMask;
    std::mutex& mutex = mutexPool->getMutex(mutexIdx);
    {
      std::lock_guard<std::mutex> lock(mutex);
      entry.ptr.swap(buf);
    }
    buf.reset();
  }
}
