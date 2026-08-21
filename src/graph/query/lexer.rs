/* hnsqr/src/graph/query/lexer.rs */
//!▫~•◦-------------------------------‣
//! # Zero-Copy GraphQuery Lexer
//!▫~•◦-------------------------------------------------------------------‣
//!
//! A hand-rolled, allocation-free lexer over a borrowed `&str` input.
//! Every `Token` variant carries a `&'src str` slice into the original
//! query string — no `String` copies are made during tokenisation.
//!
//! ## Supported token classes
//! Keywords, identifiers, labels, string literals, integer/float literals,
//! punctuation (`(`, `)`, `[`, `]`, `-`, `>`, `<`, `:`, `,`, `.`, `{`, `}`,
//! `|`, `*`, `=`, `<>`, `<=`, `>=`), and the HNSQR-specific `VECTOR MATCH`
//! extension syntax.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

// ── Token ────────────────────────────────────────────────────────────────────

/// A single lexical token.  All `&'src str` slices reference the original
/// query buffer — zero heap allocation during tokenisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Token<'src> {
    // ── Keywords ─────────────────────────────────────────────────────────
    Match,
    OptionalMatch,
    Where,
    Return,
    Limit,
    With,
    And,
    Or,
    Not,
    Is,
    Null,
    Create,
    Delete,
    Merge,
    Set,
    /// HNSQR extension: `VECTOR`
    Vector,
    /// HNSQR extension: `CERTIFIED` | `HIGH_RECALL` | `BOUNDED`
    Certified,
    HighRecall,
    Bounded,
    As,
    True,
    False,
    Shortestpath,

    // ── Identifiers / labels ─────────────────────────────────────────────
    Ident(&'src str),

    // ── Literals ─────────────────────────────────────────────────────────
    /// Quoted string literal (content without surrounding quotes).
    StringLit(&'src str),
    /// Integer literal (raw text slice).
    IntLit(&'src str),
    /// Float literal (raw text slice, contains `.`).
    FloatLit(&'src str),

    // ── Punctuation ──────────────────────────────────────────────────────
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]
    LBrace,   // {
    RBrace,   // }
    Dash,     // -
    Gt,       // >
    Lt,       // <
    Colon,    // :
    Comma,    // ,
    Dot,      // .
    Pipe,     // |
    Star,     // *
    Eq,       // =
    Ne,       // <>
    Le,       // <=
    Ge,       // >=
    Dollar,   // $
    DotDot,   // ..

    // ── Structural ───────────────────────────────────────────────────────
    Eof,
}

// ── LexError ─────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LexError {
    pub offset: usize,
    pub message: &'static str,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lex error at offset {}: {}", self.offset, self.message)
    }
}

// ── Lexer ─────────────────────────────────────────────────────────────────────

/// Zero-copy lexer over a borrowed query string.
///
/// Internally advances a byte cursor `pos` over `src`.  All tokens borrow
/// directly from `src`; no heap allocation occurs until the caller converts
/// an `Ident` or `StringLit` to owned data.
pub struct Lexer<'src> {
    src: &'src str,
    pos: usize,
}

impl<'src> Lexer<'src> {
    /// Creates a new lexer over the full query string.
    #[inline]
    pub fn new(src: &'src str) -> Self {
        Self { src, pos: 0 }
    }

    /// Returns the next token without advancing the cursor (peek).
    pub fn peek(&mut self) -> Result<Token<'src>, LexError> {
        let saved = self.pos;
        let tok = self.next_token()?;
        self.pos = saved;
        Ok(tok)
    }

    /// Advances past the next token and returns it.
    pub fn advance(&mut self) -> Result<Token<'src>, LexError> {
        self.next_token()
    }

    /// Advances and asserts the token matches `expected`.
    pub fn expect(&mut self, expected: Token<'_>) -> Result<(), LexError> {
        let tok = self.next_token()?;
        // Compare by discriminant since Token<'_> carries lifetimes.
        if tok_eq_kind(&tok, &expected) {
            Ok(())
        } else {
            Err(LexError {
                offset: self.pos,
                message: "Unexpected token",
            })
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────

    fn byte_at(&self, offset: usize) -> Option<u8> {
        self.src.as_bytes().get(self.pos + offset).copied()
    }

    /// Skips ASCII whitespace and `//`-line comments.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            // Skip ASCII whitespace.
            while self.pos < self.src.len() && self.src.as_bytes()[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            // Skip `//` line comments.
            if self.src.as_bytes().get(self.pos) == Some(&b'/')
                && self.src.as_bytes().get(self.pos + 1) == Some(&b'/')
            {
                while self.pos < self.src.len() && self.src.as_bytes()[self.pos] != b'\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn next_token(&mut self) -> Result<Token<'src>, LexError> {
        self.skip_whitespace_and_comments();

        if self.pos >= self.src.len() {
            return Ok(Token::Eof);
        }

        let b = self.src.as_bytes()[self.pos];

        match b {
            b'(' => {
                self.pos += 1;
                Ok(Token::LParen)
            }
            b')' => {
                self.pos += 1;
                Ok(Token::RParen)
            }
            b'[' => {
                self.pos += 1;
                Ok(Token::LBracket)
            }
            b']' => {
                self.pos += 1;
                Ok(Token::RBracket)
            }
            b'{' => {
                self.pos += 1;
                Ok(Token::LBrace)
            }
            b'}' => {
                self.pos += 1;
                Ok(Token::RBrace)
            }
            b',' => {
                self.pos += 1;
                Ok(Token::Comma)
            }
            b'.' => {
                // `..` for hop range
                if self.byte_at(1) == Some(b'.') {
                    self.pos += 2;
                    Ok(Token::DotDot)
                } else {
                    self.pos += 1;
                    Ok(Token::Dot)
                }
            }
            b'|' => {
                self.pos += 1;
                Ok(Token::Pipe)
            }
            b'*' => {
                self.pos += 1;
                Ok(Token::Star)
            }
            b'$' => {
                self.pos += 1;
                Ok(Token::Dollar)
            }
            b'=' => {
                self.pos += 1;
                Ok(Token::Eq)
            }
            b':' => {
                self.pos += 1;
                Ok(Token::Colon)
            }
            b'>' => {
                if self.byte_at(1) == Some(b'=') {
                    self.pos += 2;
                    Ok(Token::Ge)
                } else {
                    self.pos += 1;
                    Ok(Token::Gt)
                }
            }
            b'<' => {
                if self.byte_at(1) == Some(b'=') {
                    self.pos += 2;
                    Ok(Token::Le)
                } else if self.byte_at(1) == Some(b'>') {
                    self.pos += 2;
                    Ok(Token::Ne)
                } else {
                    self.pos += 1;
                    Ok(Token::Lt)
                }
            }
            b'-' => {
                self.pos += 1;
                Ok(Token::Dash)
            }
            // String literals: single or double quoted.
            b'\'' | b'"' => self.lex_string_literal(b),
            // Numeric literals.
            b'0'..=b'9' => self.lex_number(),
            // Identifiers / keywords.
            b'A'..=b'Z' | b'a'..=b'z' | b'_' => self.lex_ident_or_keyword(),
            // Backtick-quoted identifiers (escaped names).
            b'`' => self.lex_backtick_ident(),
            _ => Err(LexError {
                offset: self.pos,
                message: "Unexpected character",
            }),
        }
    }

    fn lex_string_literal(&mut self, quote: u8) -> Result<Token<'src>, LexError> {
        self.pos += 1; // skip opening quote
        let start = self.pos;
        loop {
            if self.pos >= self.src.len() {
                return Err(LexError {
                    offset: start - 1,
                    message: "Unterminated string literal",
                });
            }
            let c = self.src.as_bytes()[self.pos];
            if c == b'\\' {
                self.pos += 2; // skip escape sequence — we borrow as-is, caller can unescape
            } else if c == quote {
                break;
            } else {
                self.pos += 1;
            }
        }
        let s = &self.src[start..self.pos];
        self.pos += 1; // skip closing quote
        Ok(Token::StringLit(s))
    }

    fn lex_number(&mut self) -> Result<Token<'src>, LexError> {
        let start = self.pos;
        let mut is_float = false;
        while self.pos < self.src.len() {
            match self.src.as_bytes()[self.pos] {
                b'0'..=b'9' => self.pos += 1,
                b'.' if !is_float && self.byte_at(1) != Some(b'.') => {
                    is_float = true;
                    self.pos += 1;
                }
                _ => break,
            }
        }
        let s = &self.src[start..self.pos];
        if is_float {
            Ok(Token::FloatLit(s))
        } else {
            Ok(Token::IntLit(s))
        }
    }

    fn lex_ident_or_keyword(&mut self) -> Result<Token<'src>, LexError> {
        let start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src.as_bytes()[self.pos];
            if c.is_ascii_alphanumeric() || c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        let word = &self.src[start..self.pos];
        Ok(keyword_or_ident(word))
    }

    fn lex_backtick_ident(&mut self) -> Result<Token<'src>, LexError> {
        self.pos += 1; // skip opening backtick
        let start = self.pos;
        while self.pos < self.src.len() && self.src.as_bytes()[self.pos] != b'`' {
            self.pos += 1;
        }
        if self.pos >= self.src.len() {
            return Err(LexError {
                offset: start - 1,
                message: "Unterminated backtick identifier",
            });
        }
        let s = &self.src[start..self.pos];
        self.pos += 1; // skip closing backtick
        Ok(Token::Ident(s))
    }
}

// ── Keyword resolution (zero-copy, case-insensitive) ─────────────────────────

/// Resolves a raw word slice to a keyword token or falls back to `Ident`.
/// Case-insensitive: `MATCH`, `match`, `Match` all produce `Token::Match`.
fn keyword_or_ident(word: &str) -> Token<'_> {
    // A 32-byte stack buffer is enough for any GraphQuery keyword.
    let mut buf = [0u8; 32];
    let len = word.len().min(buf.len());
    for (i, &b) in word.as_bytes()[..len].iter().enumerate() {
        buf[i] = b.to_ascii_uppercase();
    }
    let upper = std::str::from_utf8(&buf[..len]).unwrap_or(word);

    match upper {
        "MATCH" => Token::Match,
        "OPTIONAL" => {
            // Will be combined with MATCH by the parser.
            Token::OptionalMatch
        }
        "WHERE" => Token::Where,
        "RETURN" => Token::Return,
        "LIMIT" => Token::Limit,
        "WITH" => Token::With,
        "AND" => Token::And,
        "OR" => Token::Or,
        "NOT" => Token::Not,
        "IS" => Token::Is,
        "NULL" => Token::Null,
        "CREATE" => Token::Create,
        "DELETE" => Token::Delete,
        "MERGE" => Token::Merge,
        "SET" => Token::Set,
        "VECTOR" => Token::Vector,
        "CERTIFIED" => Token::Certified,
        "HIGH_RECALL" => Token::HighRecall,
        "BOUNDED" => Token::Bounded,
        "AS" => Token::As,
        "TRUE" => Token::True,
        "FALSE" => Token::False,
        "SHORTESTPATH" => Token::Shortestpath,
        _ => Token::Ident(word),
    }
}

/// Structural token equality ignoring lifetime and payload.
/// Used by `expect()` to check token kind without caring about value.
pub fn tok_eq_kind(a: &Token<'_>, b: &Token<'_>) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}
