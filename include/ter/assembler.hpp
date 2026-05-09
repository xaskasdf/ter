#pragma once
#include <ter/word.hpp>
#include <ter/isa.hpp>
#include <vector>
#include <string>
#include <stdexcept>

namespace ter {

class AssemblerError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::vector<Word27> assemble(const std::string& source);

namespace detail {
struct Token {
    enum class Kind { Mnemonic, Reg, VecReg, AccReg, Number, LabelDef, Newline, EOFTok } kind;
    std::string text;
};
std::vector<Token> tokenise(const std::string& source);
}

}
