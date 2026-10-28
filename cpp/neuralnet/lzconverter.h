#ifndef LZCONVERTER_H
#define LZCONVERTER_H

#include <string>

struct ModelDesc;

//Converts Leela Zero / ELF text weights (v1/v2, plain 19x19 SINGLE-head residual nets)
//directly into a ModelDesc, so that KataGo can load LZ weights without an offline
//Python conversion step or a text round-trip through the model parser.
namespace LZConverter {

  //Returns true if the (already decompressed) file contents look like LZ weights:
  //first token is an integer 1 or 2 and the following lines match the standard
  //INPUT_CONV / RESCONV layout. Purely a cheap structural pre-check.
  bool looksLikeLZ(const std::string& data);

  //Parses the LZ weights and fills desc with an equivalent mv18 ModelDesc.
  //Throws StringError if the net is not a supported standard LZ v1/v2 net.
  void buildModelDesc(const std::string& data, const std::string& sha256, ModelDesc& desc);

}

#endif
