grammar SimpleLang;

program
  : statement* EOF
  ;

statement
  : varDecl
  | printStmt
  | ifStmt
  ;

varDecl
  : NUMBER_KW ID '=' expr NEWLINE*
  | TEXT_KW   ID '=' expr NEWLINE*
  ;

printStmt
  : PRINT expr NEWLINE*
  ;

ifStmt
  : IF condition block (ELSE block)? NEWLINE*
  ;

block
  : LBRACE NEWLINE* statement* RBRACE
  ;

condition
  : expr compOp expr
  ;

compOp
  : '==' | '!=' | '<' | '<=' | '>' | '>='
  ;

expr
  : expr op=('*'|'/') expr       # MulDiv
  | expr op=('+'|'-') expr       # AddSub
  | INT                          # IntLit
  | STRING                       # StringLit
  | ID                           # VarRef
  | '(' expr ')'                 # Parens
  ;

NUMBER_KW : 'number';
TEXT_KW   : 'text';
PRINT     : 'print';
IF        : 'if';
ELSE      : 'else';

ID        : [a-zA-Z_][a-zA-Z0-9_]*;
INT       : [0-9]+;
STRING    : '"' ( '\\' . | ~["\\] )* '"';

LBRACE    : '{';
RBRACE    : '}';

NEWLINE   : ('\r'? '\n')+;
WS        : [ \t]+ -> skip;
COMMENT   : '//' ~[\r\n]* -> skip;
