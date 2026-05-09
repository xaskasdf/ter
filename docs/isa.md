# ter ISA — preliminary reference (F0-F2)

## Registers
- R0..R26 — 27 scalar Word27 registers. R0 is hard-zero. R26 is stack pointer by convention.
- V0..V8  — 9 vector registers, 27 lanes of 9-trit values.
- A0..A2  — 3 accumulator registers (Word54-equivalent).
- PC, halt-flag.

## Instruction format
27-trit fixed: [opcode 6t] [dst 3t] [src1 3t] [src2 3t] [imm 12t].

## Opcodes (current)
| Mnemonic | Code | Operands | Semantics |
|---|---|---|---|
| tnop  | 0  | — | no-op |
| thalt | 1  | — | halt |
| tdbg  | 2  | — | debug trap |
| tadd  | 10 | rd, rs1, rs2 | rd = rs1 + rs2 |
| tsub  | 11 | rd, rs1, rs2 | rd = rs1 - rs2 |
| tneg  | 12 | rd, rs1      | rd = -rs1 |
| tabs  | 13 | rd, rs1      | rd = abs(rs1) |
| tand3 | 14 | rd, rs1, rs2 | per-trit min |
| tor3  | 15 | rd, rs1, rs2 | per-trit max |
| txor3 | 16 | rd, rs1, rs2 | consensus |
| tcmp  | 17 | rd, rs1, rs2 | rd[0] = sign(rs1 - rs2) |
| tsign | 18 | rd, rs1      | rd[0] = sign(rs1) |
| tload   | 30 | rd, rs1      | rd = mem[rs1] |
| tstore  | 31 | rs1, rs2     | mem[rs2] = rs1 |
| tloadi  | 32 | rd, imm      | rd = imm |
| tbeq    | 50 | rs1, rs2, imm| if rs1==rs2 PC=imm |
| tbne    | 51 | rs1, rs2, imm| if rs1!=rs2 PC=imm |
| tblt    | 52 | rs1, rs2, imm| if rs1<rs2 PC=imm |
| tjump   | 53 | imm          | PC=imm |
| tcall   | 54 | imm          | push PC, PC=imm; SP at R26 |
| tret    | 55 | —            | pop PC; SP at R26 |
| tvadd   | 100 | vd, vs1, vs2 | per-lane add |
| tvsub   | 101 | vd, vs1, vs2 | per-lane sub |
| tvneg   | 102 | vd, vs1      | per-lane neg |
| tvbroadcast | 103 | vd, imm | each lane = imm |
| tvmac   | 110 | ad, vs1, vs2 | acc += vs1 * vs2 lane-wise |
| tvsum   | 111 | rd, as1      | rd = sum lanes(as1) |
| tvmax   | 112 | vd, vs1      | per-lane running max |
| tvshuf  | 113 | vd, vs1, imm | rotate lanes by imm |
| tvload  | 120 | vd, rs1      | load 27 words at mem[rs1] |
| tvstore | 121 | vs1, rs2     | store vs1 at mem[rs2] |

## Calling convention (F1+)
- R1..R7 — argument and return.
- R8..R15 — caller-saved scratch.
- R16..R25 — callee-saved.
- R26 — stack pointer (post-increment on push).

Full ISA spec lives in the design at `docs/superpowers/specs/2026-05-08-ter-design.md`.
