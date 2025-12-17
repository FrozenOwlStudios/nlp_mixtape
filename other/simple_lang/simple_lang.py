from dataclasses import dataclass
from typing import Any, Dict

from antlr4 import FileStream, CommonTokenStream, InputStream
from antlr4.error.ErrorListener import ErrorListener

from antlr_out.SimpleLangLexer import SimpleLangLexer
from antlr_out.SimpleLangParser import SimpleLangParser
from antlr_out.SimpleLangVisitor import SimpleLangVisitor


class ThrowingErrorListener(ErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        raise SyntaxError(f"line {line}:{column} {msg}")

# antlr4 -Dlanguage=Python3 Query.g4 -visitor -o folder

@dataclass
class Value:
    kind: str  # "number" or "text"
    value: Any


class Interpreter(SimpleLangVisitor):
    def __init__(self) -> None:
        self.env: Dict[str, Value] = {}

    # program: statement* EOF
    def visitProgram(self, ctx: SimpleLangParser.ProgramContext):
        for st in ctx.statement():
            self.visit(st)
        return None

    # --- statements ---

    def visitVarDecl(self, ctx: SimpleLangParser.VarDeclContext):
        name = ctx.ID().getText()
        expr_val = self.visit(ctx.expr())

        if ctx.NUMBER_KW() is not None:
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
            if ctx.ELSE() is not None:
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

        op = ctx.compOp().getText()
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
    def visitIntLit(self, ctx: SimpleLangParser.IntLitContext) -> Value:
        return Value("number", int(ctx.INT().getText()))

    def visitStringLit(self, ctx: SimpleLangParser.StringLitContext) -> Value:
        raw = ctx.STRING().getText()  # includes quotes
        # simple unescape using python parsing rules for escape sequences
        s = bytes(raw[1:-1], "utf-8").decode("unicode_escape")
        return Value("text", s)

    def visitVarRef(self, ctx: SimpleLangParser.VarRefContext) -> Value:
        name = ctx.ID().getText()
        if name not in self.env:
            raise NameError(f"Undefined variable '{name}'")
        return self.env[name]

    def visitParens(self, ctx: SimpleLangParser.ParensContext) -> Value:
        return self.visit(ctx.expr())

    def visitMulDiv(self, ctx: SimpleLangParser.MulDivContext) -> Value:
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        if left.kind != "number" or right.kind != "number":
            raise TypeError("'*' and '/' only work on numbers")

        op = ctx.op.text
        if op == "*":
            return Value("number", left.value * right.value)
        if op == "/":
            return Value("number", left.value / right.value)
        raise RuntimeError("Unexpected operator")

    def visitAddSub(self, ctx: SimpleLangParser.AddSubContext) -> Value:
        left = self.visit(ctx.expr(0))
        right = self.visit(ctx.expr(1))
        op = ctx.op.text

        if left.kind != "number" or right.kind != "number":
            raise TypeError("'+' and '-' only work on numbers (no string concat in this example)")

        if op == "+":
            return Value("number", left.value + right.value)
        if op == "-":
            return Value("number", left.value - right.value)
        raise RuntimeError("Unexpected operator")


def run(source: str) -> None:
    lexer = SimpleLangLexer(InputStream(source))
    stream = CommonTokenStream(lexer)
    parser = SimpleLangParser(stream)

    lexer.removeErrorListeners()
    parser.removeErrorListeners()
    lexer.addErrorListener(ThrowingErrorListener())
    parser.addErrorListener(ThrowingErrorListener())

    tree = parser.program()
    interp = Interpreter()
    interp.visit(tree)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python interpreter.py <file.sl>")
        sys.exit(2)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        run(f.read())
