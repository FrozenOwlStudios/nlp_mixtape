
# ==============================================================================
#                                IMPORTS
# ==============================================================================

# Python std lib
import sys
from dataclasses import dataclass
from typing import Any, Dict

# ANTLR 4 general imports
from antlr4 import FileStream, CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

# ANTL 4 generated grammar
from antlr_grammar.SimpleLangLexer import SimpleLangLexer
from antlr_grammar.SimpleLangParser import SimpleLangParser
from antlr_grammar.SimpleLangVisitor import SimpleLangVisitor

# ==============================================================================
#                                INTERPRETER
# ==============================================================================
class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"line {line}:{column} {msg}")


@dataclass
class Variable:
    kind: str  # "number" or "text"
    value: Any

class SimpleLangInterpreter(SimpleLangVisitor):
    def __init__(self) -> None:
        self.env: Dict[str, Variable] = {}

    # program: statement* EOF
    def visitProgram(self, ctx: SimpleLangParser.ProgramContext):
        for st in ctx.statement():
            self.visit(st)
        return None

    # --- statements ---

    def visitVarDecl(self, ctx: SimpleLangParser.VarDeclContext):
        name = ctx.ID().getText()
        expr_val = self.visit(ctx.expr())

        if ctx.KW_NUMBER() is not None:
            if expr_val.kind != "number":
                raise TypeError(f"Variable '{name}' declared as number but assigned {expr_val.kind}")
            self.env[name] = expr_val
        else:
            # text
            if expr_val.kind != "text":
                raise TypeError(f"Variable '{name}' declared as text but assigned {expr_val.kind}")
            self.env[name] = expr_val

        return None

    def visitPrintStmt(self, ctx: SimpleLangParser.PrintStmtContext):
        v = self.visit(ctx.expr())
        print(v.value)
        return None

    def visitIfStmt(self, ctx: SimpleLangParser.IfStmtContext):
        cond = self.visit(ctx.condition())
        if cond:
            self.visit(ctx.block(0))
        else:
            if ctx.KW_ELSE() is not None:
                self.visit(ctx.block(1))
        return None

    def visitBlock(self, ctx: SimpleLangParser.BlockContext):
        for st in ctx.statement():
            self.visit(st)
        return None

    # --- condition ---
    def visitCondition(self, ctx: SimpleLangParser.ConditionContext) -> bool:
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if left.kind != "number" or right.kind != "number":
            raise TypeError("if condition comparisons require numbers")

        op = ctx.OP_COMP().getText()
        a = left.value
        b = right.value

        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        if op == "<":
            return a < b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        if op == ">=":
            return a >= b
        raise RuntimeError(f"Unknown comparison operator: {op}")

    # --- expressions ---
    def visitIntLit(self, ctx: SimpleLangParser.IntLitContext) -> Variable:
        return Variable("number", int(ctx.INT().getText()))

    def visitStringLit(self, ctx: SimpleLangParser.StringLitContext) -> Variable:
        raw = ctx.STRING().getText()  # includes quotes
        # simple unescape using python parsing rules for escape sequences
        s = bytes(raw[1:-1], "utf-8").decode("unicode_escape")
        return Variable("text", s)

    def visitVarRef(self, ctx: SimpleLangParser.VarRefContext) -> Variable:
        name = ctx.ID().getText()
        if name not in self.env:
            raise NameError(f"Undefined variable '{name}'")
        return self.env[name]

    def visitParens(self, ctx: SimpleLangParser.ParensContext) -> Variable:
        return self.visit(ctx.expr())

    def visitArithmOp(self, ctx: SimpleLangParser.ArithmOpContext) -> Variable:
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if left.kind != "number" or right.kind != "number":
            raise TypeError("'*' and '/' only work on numbers")

        op = ctx.OP_ARITHM().getText()
        if op == '+':
            return Variable("number", left.value + right.value)
        if op == '-':
            return Variable("number", left.value - right.value)
        if op == '*':
            return Variable("number", left.value * right.value)
        if op == '/':
            return Variable("number", left.value / right.value)
        
        raise RuntimeError(f"Unexpected operator {op}")

# ==============================================================================
#                                MAIN
# ==============================================================================
def run_interpreter(source: str) -> None:
    lexer = SimpleLangLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    parser = SimpleLangParser(stream)

    lexer.removeErrorListeners()
    parser.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener())
    parser.addErrorListener(ThrowingErrorListener())

    tree = parser.program()
    interp = SimpleLangInterpreter()
    interp.visit(tree)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python interpreter.py <file.sl>")
        sys.exit(2)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        run_interpreter(f.read())