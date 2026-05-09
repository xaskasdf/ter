#include <ter/assembler.hpp>
#include <unordered_map>
#include <stdexcept>

namespace ter {

std::vector<Word27> assemble(const std::string& source) {
    using namespace detail;
    auto toks = tokenise(source);

    static const std::unordered_map<std::string, Opcode> mnem = {
        {"tnop", Opcode::TNOP},   {"thalt", Opcode::THALT}, {"tdbg", Opcode::TDBG},
        {"tadd", Opcode::TADD},   {"tsub",  Opcode::TSUB},  {"tneg", Opcode::TNEG},
        {"tabs", Opcode::TABS},   {"tand3", Opcode::TAND3}, {"tor3", Opcode::TOR3},
        {"txor3", Opcode::TXOR3}, {"tcmp",  Opcode::TCMP},  {"tsign", Opcode::TSIGN},
        {"tload", Opcode::TLOAD}, {"tstore",Opcode::TSTORE},{"tloadi",Opcode::TLOADI},
        {"tbeq",  Opcode::TBEQ},  {"tbne",  Opcode::TBNE},  {"tblt", Opcode::TBLT},
        {"tjump", Opcode::TJUMP}, {"tcall", Opcode::TCALL}, {"tret", Opcode::TRET},
    };

    std::unordered_map<std::string, int> labels;
    std::vector<std::vector<Token>> lines;
    std::vector<Token> current;
    int instr_idx = 0;
    auto flush = [&] {
        if (!current.empty()) {
            size_t k = 0;
            while (k < current.size() && current[k].kind == Token::Kind::LabelDef) {
                labels[current[k].text] = instr_idx;
                ++k;
            }
            if (k < current.size()) {
                std::vector<Token> t(current.begin() + static_cast<ptrdiff_t>(k), current.end());
                lines.push_back(std::move(t));
                ++instr_idx;
            }
            current.clear();
        }
    };
    for (const auto& t : toks) {
        if (t.kind == Token::Kind::Newline || t.kind == Token::Kind::EOFTok) flush();
        else current.push_back(t);
    }

    std::vector<Word27> blob;
    blob.reserve(lines.size());
    for (const auto& line : lines) {
        if (line.empty() || line[0].kind != Token::Kind::Mnemonic)
            throw AssemblerError("expected mnemonic");
        auto it = mnem.find(line[0].text);
        if (it == mnem.end()) throw AssemblerError("unknown mnemonic: " + line[0].text);
        Instr i{};
        i.op = it->second;

        auto reg_at = [&](size_t k) -> uint8_t {
            if (k >= line.size() || line[k].kind != Token::Kind::Reg)
                throw AssemblerError("expected scalar register");
            int n = std::stoi(line[k].text);
            if (n < 0 || n > 26) throw AssemblerError("register out of range");
            return static_cast<uint8_t>(n);
        };
        auto imm_at = [&](size_t k) -> int32_t {
            if (k >= line.size()) throw AssemblerError("expected immediate or label");
            const auto& tk = line[k];
            if (tk.kind == Token::Kind::Number) return std::stoi(tk.text);
            if (tk.kind == Token::Kind::Mnemonic) {
                auto lit = labels.find(tk.text);
                if (lit == labels.end()) throw AssemblerError("undefined label: " + tk.text);
                return lit->second;
            }
            throw AssemblerError("expected immediate");
        };

        switch (i.op) {
            case Opcode::TNOP: case Opcode::THALT: case Opcode::TDBG: case Opcode::TRET:
                break;
            case Opcode::TLOADI:
                i.dst = reg_at(1); i.imm = imm_at(2); break;
            case Opcode::TADD: case Opcode::TSUB: case Opcode::TAND3:
            case Opcode::TOR3: case Opcode::TXOR3: case Opcode::TCMP:
                i.dst = reg_at(1); i.src1 = reg_at(2); i.src2 = reg_at(3); break;
            case Opcode::TNEG: case Opcode::TABS: case Opcode::TSIGN:
                i.dst = reg_at(1); i.src1 = reg_at(2); break;
            case Opcode::TLOAD:
                i.dst = reg_at(1); i.src1 = reg_at(2); break;
            case Opcode::TSTORE:
                i.src1 = reg_at(1); i.src2 = reg_at(2); break;
            case Opcode::TBEQ: case Opcode::TBNE: case Opcode::TBLT:
                i.src1 = reg_at(1); i.src2 = reg_at(2); i.imm = imm_at(3); break;
            case Opcode::TJUMP: case Opcode::TCALL:
                i.imm = imm_at(1); break;
            default:
                throw AssemblerError("unsupported in F1.8 emitter");
        }

        blob.push_back(encode(i));
    }
    return blob;
}

}
