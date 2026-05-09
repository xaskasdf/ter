#include <ter/assembler.hpp>
#include <sstream>
#include <cctype>
#include <algorithm>

namespace ter::detail {

std::vector<Token> tokenise(const std::string& source) {
    std::vector<Token> out;
    std::istringstream is(source);
    std::string line;
    while (std::getline(is, line)) {
        auto sc = line.find(';');
        if (sc != std::string::npos) line.erase(sc);
        size_t i = 0;
        while (i < line.size()) {
            if (std::isspace(static_cast<unsigned char>(line[i]))) { ++i; continue; }
            if (line[i] == ',') { ++i; continue; }
            size_t j = i;
            while (j < line.size() && !std::isspace(static_cast<unsigned char>(line[j])) && line[j] != ',') ++j;
            std::string tok = line.substr(i, j - i);
            i = j;
            if (!tok.empty() && tok.back() == ':') {
                out.push_back({Token::Kind::LabelDef, tok.substr(0, tok.size() - 1)});
            } else if (tok.size() >= 2 &&
                       (tok[0] == 'r' || tok[0] == 'R' || tok[0] == 'v' || tok[0] == 'V' ||
                        tok[0] == 'a' || tok[0] == 'A') &&
                       std::all_of(tok.begin() + 1, tok.end(),
                                   [](char c){ return std::isdigit(static_cast<unsigned char>(c)); })) {
                char first = tok[0];
                auto kind = (first=='v'||first=='V') ? Token::Kind::VecReg
                          :(first=='a'||first=='A') ? Token::Kind::AccReg
                          :                            Token::Kind::Reg;
                out.push_back({kind, tok.substr(1)});
            } else if (!tok.empty() && (std::isdigit(static_cast<unsigned char>(tok[0])) ||
                                        tok[0] == '-' || tok[0] == '+')) {
                out.push_back({Token::Kind::Number, tok});
            } else {
                out.push_back({Token::Kind::Mnemonic, tok});
            }
        }
        out.push_back({Token::Kind::Newline, ""});
    }
    out.push_back({Token::Kind::EOFTok, ""});
    return out;
}

}
