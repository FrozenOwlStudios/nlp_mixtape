grammar SimpleLang;

// antlr4 -Dlanguage=Python3 Query.g4 -visitor -o folder

//=================================================================================================
//                                               PARSER
//=================================================================================================


program
  : statement* EOF
  ;

statement
  : varDecl
  | printStmt
  | ifStmt
  ;

varDecl
  : KW_NUMBER ID OP_ASSIGN expr NEWLINE*
  | KW_TEXT   ID OP_ASSIGN expr NEWLINE*
  ;

printStmt
  : KW_PRINT expr NEWLINE*
  ;

ifStmt
  : KW_IF condition block (KW_ELSE block)? NEWLINE*
  ;

block
  : BLOCK_START NEWLINE* statement* BLOCK_END
  ;

condition
  : expr OP_COMP expr
  ;


expr
  : expr OP_ARITHM expr     
  | INT                          
  | STRING                       
  | ID                          
  | PAREN_START expr PAREN_END 
  ;

//=================================================================================================
//                                               LEXER
//=================================================================================================

// Keywords
KW_NUMBER : 'number';
KW_TEXT  : 'text';
KW_PRINT     : 'print';
KW_IF        : 'if';
KW_ELSE      : 'else';

// Operators
OP_COMP : '==' | '!=' | '<' | '<=' | '>' | '>=';
OP_ARITHM : '+'|'-'|'*'|'/';
OP_ASSIGN: '=';

// Acceptable variable names
ID        : [a-zA-Z_][a-zA-Z0-9_]*;

// Acceptable variable values
INT       : [0-9]+;
STRING    : '"' ( '\\' . | ~["\\] )* '"';

// Things to ignore
WS        : [ \t]+ -> skip;
COMMENT   : '//' ~[\r\n]* -> skip;

// Scope control
BLOCK_START    : '{';
BLOCK_END    : '}';
PAREN_START : '(';
PAREN_END : ')';

// Other stuff
NEWLINE   : ('\r'? '\n')+;
