/* hnsqr/src/graph/query/parser.rs */
//!▫~•◦-------------------------------‣
//! # Zero-Copy GraphQuery Parser
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Converts a `&str` GraphQuery query directly into a [`QueryAst`] using the
//! zero-copy [`Lexer`].  No heap allocation occurs during tokenisation;
//! the only allocations are the final `String` fields inside `QueryAst`
//! (one per unique identifier / literal).
//!
//! ## Supported grammar (HNSQR Graph Query Profile v1)
//!
//! ```text
//! Query         ::= VectorMatch? MatchClause* WhereClause? ReturnClause
//!                 | MutationQuery
//!
//! VectorMatch   ::= 'VECTOR' 'MATCH' '(' Alias ')' 'USING' '$'param
//!                   'LIMIT' k 'CONTRACT' Contract
//!
//! MatchClause   ::= 'MATCH' Pattern (',' Pattern)*
//!                 | 'OPTIONAL' 'MATCH' Pattern (',' Pattern)*
//!
//! Pattern       ::= NodeSpec (RelSpec NodeSpec)*
//! NodeSpec      ::= '(' Alias? (':' Label)? PropFilter? ')'
//! RelSpec       ::= '<'? '-' '[' Alias? (':' RelType)? ('*' HopRange?)? ']' '-' '>'?
//!
//! HopRange      ::= IntLit ('..' IntLit)?
//!
//! WhereClause   ::= 'WHERE' Predicate
//! Predicate     ::= AndPred ('OR' AndPred)*
//! AndPred       ::= NotPred ('AND' NotPred)*
//! NotPred       ::= 'NOT' NotPred | Atom
//! Atom          ::= PropRef Op Value
//!                 | PropRef 'IS' 'NULL'
//!                 | PropRef 'IS' 'NOT' 'NULL'
//!                 | '(' Predicate ')'
//!
//! ReturnClause  ::= 'RETURN' ReturnItem (',' ReturnItem)* ('LIMIT' Int)?
//! ReturnItem    ::= Alias ('.' Key)?
//!
//! MutationQuery ::= 'CREATE' NodeSpec+
//!                 | 'DELETE' Alias (',' Alias)*
//! ```
//!
//! ## Zero-copy invariant
//! The `Lexer` operates over `&'src str` slices.  All identifier strings
//! remain as borrows until the parser calls `.to_string()` exactly once per
//! unique name when building the owned `QueryAst`.  No intermediate `String`
//! buffers are used.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::catalog::labels::LabelCatalog;
use crate::graph::catalog::relationships::RelTypeCatalog;
use crate::graph::query::ast::{
    Direction, GraphMutationClause, GraphPattern, PredicateValue, QueryAst, ReturnClause,
    ReturnItem, ScalarPredicate, VectorContract, VectorMatchClause, WhereClause,
};
use crate::graph::query::lexer::{LexError, Lexer, Token};

// ── ParseError ───────────────────────────────────────────────────────────────

/// Errors produced by the GraphQuery parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// Wrapped lexer error.
    Lex(LexError),
    /// Unexpected token encountered.
    UnexpectedToken { offset: usize, msg: String },
    /// Semantically invalid construct.
    Invalid(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lex(e) => write!(f, "Lex error: {e}"),
            Self::UnexpectedToken { offset, msg } => {
                write!(f, "Parse error at offset {offset}: {msg}")
            }
            Self::Invalid(s) => write!(f, "Invalid query: {s}"),
        }
    }
}

impl std::error::Error for ParseError {}

impl From<LexError> for ParseError {
    fn from(e: LexError) -> Self {
        Self::Lex(e)
    }
}

// ── Parser ───────────────────────────────────────────────────────────────────

/// Recursive-descent GraphQuery parser.
///
/// Holds references to the label and relationship-type catalogs so that
/// `Label` and `RelType` names are resolved to compact IDs at parse time,
/// avoiding a separate catalog-resolution pass.
pub struct Parser<'src> {
    lex: Lexer<'src>,
    label_catalog: &'src LabelCatalog,
    rel_catalog: &'src RelTypeCatalog,
}

impl<'src> Parser<'src> {
    /// Creates a new parser over `src` using the provided catalogs.
    pub fn new(
        src: &'src str,
        label_catalog: &'src LabelCatalog,
        rel_catalog: &'src RelTypeCatalog,
    ) -> Self {
        Self {
            lex: Lexer::new(src),
            label_catalog,
            rel_catalog,
        }
    }

    /// Parses the full query and returns a [`QueryAst`].
    pub fn parse(mut self) -> Result<QueryAst, ParseError> {
        let mut ast = QueryAst {
            vector_match: None,
            patterns: Vec::new(),
            where_clause: WhereClause::default(),
            return_clause: ReturnClause { items: Vec::new(), limit: None },
            mutations: Vec::new(),
            unwind: None,
            subqueries: Vec::new(),
        };

        loop {
            match self.lex.peek()? {
                Token::Eof => break,
                Token::Vector => {
                    ast.vector_match = Some(self.parse_vector_match()?);
                }
                Token::Match => {
                    self.lex.advance()?;
                    self.parse_match_patterns(&mut ast.patterns, false)?;
                }
                Token::OptionalMatch => {
                    // OPTIONAL keyword — next must be MATCH
                    self.lex.advance()?;
                    match self.lex.advance()? {
                        Token::Match => {}
                        _ => return Err(self.err("Expected MATCH after OPTIONAL")),
                    }
                    self.parse_match_patterns(&mut ast.patterns, true)?;
                }
                Token::Where => {
                    self.lex.advance()?;
                    ast.where_clause.predicates.push(self.parse_predicate()?);
                }
                Token::Return => {
                    self.lex.advance()?;
                    ast.return_clause = self.parse_return_clause()?;
                }
                Token::Create => {
                    self.lex.advance()?;
                    ast.mutations.extend(self.parse_create_clause()?);
                }
                Token::Delete => {
                    self.lex.advance()?;
                    ast.mutations.extend(self.parse_delete_clause()?);
                }
                _ => {
                    // Skip unrecognised top-level tokens gracefully.
                    self.lex.advance()?;
                }
            }
        }

        Ok(ast)
    }

    // ── VECTOR MATCH ─────────────────────────────────────────────────────

    /// Parses: `VECTOR MATCH (alias) USING $param LIMIT k [CONTRACT contract]`
    fn parse_vector_match(&mut self) -> Result<VectorMatchClause, ParseError> {
        self.lex.advance()?; // consume VECTOR
        // Next must be MATCH
        match self.lex.advance()? {
            Token::Match => {}
            _ => return Err(self.err("Expected MATCH after VECTOR")),
        }

        self.expect_lparen()?;
        let binding = self.expect_ident()?;
        self.expect_rparen()?;

        // USING $param
        match self.lex.peek()? {
            Token::Ident(s) if s.eq_ignore_ascii_case("USING") => {
                self.lex.advance()?;
            }
            _ => {} // USING is optional — next token should be $ or LIMIT
        }

        let query_param = if let Token::Dollar = self.lex.peek()? {
            self.lex.advance()?; // consume $
            self.expect_ident()?
        } else {
            binding.clone()
        };

        // LIMIT k
        let k = match self.lex.peek()? {
            Token::Limit => {
                self.lex.advance()?;
                self.expect_uint()?
            }
            _ => 10, // sensible default
        };

        // CONTRACT Certified | HIGH_RECALL | BOUNDED
        let contract = match self.lex.peek()? {
            Token::Ident(s) if s.eq_ignore_ascii_case("CONTRACT") => {
                self.lex.advance()?;
                match self.lex.advance()? {
                    Token::Certified | Token::Ident(_) => VectorContract::Certified,
                    Token::HighRecall => VectorContract::HighRecall,
                    Token::Bounded => VectorContract::Bounded,
                    _ => VectorContract::Certified,
                }
            }
            Token::Certified => {
                self.lex.advance()?;
                VectorContract::Certified
            }
            Token::HighRecall => {
                self.lex.advance()?;
                VectorContract::HighRecall
            }
            Token::Bounded => {
                self.lex.advance()?;
                VectorContract::Bounded
            }
            _ => VectorContract::Certified,
        };

        Ok(VectorMatchClause { binding, query_param, k, contract })
    }

    // ── MATCH pattern list ────────────────────────────────────────────────

    /// Parses one or more comma-separated patterns after a MATCH keyword.
    fn parse_match_patterns(
        &mut self,
        out: &mut Vec<GraphPattern>,
        optional: bool,
    ) -> Result<(), ParseError> {
        loop {
            self.parse_pattern_chain(out, optional)?;

            // Consume trailing comma → continue; otherwise stop.
            if let Token::Comma = self.lex.peek()? {
                self.lex.advance()?;
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Parses a single path pattern: `(a:L)-[r:T]->(b)<-[s:U]-(c)` etc.
    ///
    /// Emits one `NodePattern` per node and one `Expand`/`OptionalExpand`
    /// per relationship step into `out`.
    fn parse_pattern_chain(
        &mut self,
        out: &mut Vec<GraphPattern>,
        optional: bool,
    ) -> Result<(), ParseError> {
        // First node.
        let (first_alias, first_label, first_preds) = self.parse_node_spec()?;
        if let Some(alias) = &first_alias {
            out.push(GraphPattern::NodePattern {
                alias: alias.clone(),
                label: first_label,
                predicates: first_preds,
            });
        }

        // Zero or more `-[r:T]->(b)` continuations.
        loop {
            // A relationship starts with `<-` or `-`.
            let direction_start = match self.lex.peek()? {
                Token::Dash => Direction::Outgoing,  // will refine after `]`
                Token::Lt => Direction::Incoming,     // `<-[...]-`
                _ => break,
            };

            // Parse the direction prefix.
            let incoming_prefix = if direction_start == Direction::Incoming {
                self.lex.advance()?; // consume `<`
                // must be followed by `-`
                match self.lex.advance()? {
                    Token::Dash => true,
                    _ => return Err(self.err("Expected `-` after `<`")),
                }
            } else {
                self.lex.advance()?; // consume `-`
                false
            };

            // `[r:TYPE*min..max]` or `[]` or absent (plain `-`).
            let (rel_alias, rel_type, min_hops, max_hops) =
                if let Token::LBracket = self.lex.peek()? {
                    self.lex.advance()?; // consume `[`
                    let r = self.parse_rel_spec()?;
                    r
                } else {
                    // Bare `-` or `->` with no bracket.
                    (None, None, 1u8, 1u8)
                };

            // Direction suffix: `-` or `->`.
            let outgoing_suffix = match self.lex.peek()? {
                Token::Dash => {
                    self.lex.advance()?;
                    if let Token::Gt = self.lex.peek()? {
                        self.lex.advance()?;
                        true
                    } else {
                        false
                    }
                }
                Token::Gt => {
                    self.lex.advance()?; // consume `>`
                    true
                }
                _ => false,
            };

            // Determine canonical direction.
            let direction = match (incoming_prefix, outgoing_suffix) {
                (true, false) => Direction::Incoming,
                (false, true) => Direction::Outgoing,
                _ => Direction::Undirected,
            };

            // Target node.
            let (dst_alias, dst_label, dst_preds) = self.parse_node_spec()?;
            let dst_alias_str = dst_alias.clone().unwrap_or_else(|| "_anon".to_string());

            // Emit destination NodePattern if it has a label or predicates.
            if dst_label.is_some() || !dst_preds.is_empty() {
                out.push(GraphPattern::NodePattern {
                    alias: dst_alias_str.clone(),
                    label: dst_label,
                    predicates: dst_preds,
                });
            }

            let src_alias = first_alias.clone().unwrap_or_else(|| "_anon".to_string());

            // Determine the source of this expand from the previously declared binding.
            // For chained patterns like (a)-[]->(b)-[]->(c), the src of the second
            // expand must be the dst alias from the previous expand.
            let actual_src = if out.len() >= 2 {
                // Last emitted pattern — extract the most recently declared dst alias.
                match out.last() {
                    Some(GraphPattern::Expand { dst_alias, .. })
                    | Some(GraphPattern::OptionalExpand { dst_alias, .. }) => dst_alias.clone(),
                    Some(GraphPattern::NodePattern { alias, .. }) => alias.clone(),
                    None => src_alias.clone(),
                }
            } else {
                src_alias
            };

            if optional {
                out.push(GraphPattern::OptionalExpand {
                    src_alias: actual_src,
                    rel_alias,
                    rel_type,
                    dst_alias: dst_alias_str,
                    direction,
                    min_hops,
                    max_hops,
                });
            } else {
                out.push(GraphPattern::Expand {
                    src_alias: actual_src,
                    rel_alias,
                    rel_type,
                    dst_alias: dst_alias_str,
                    direction,
                    min_hops,
                    max_hops,
                });
            }
        }

        Ok(())
    }

    /// Parses `(alias?:Label? {props}?)` and returns `(alias, label_id, predicates)`.
    fn parse_node_spec(
        &mut self,
    ) -> Result<(Option<String>, Option<u32>, Vec<ScalarPredicate>), ParseError> {
        self.expect_lparen()?;

        let alias = match self.lex.peek()? {
            Token::Ident(_) => Some(self.expect_ident()?),
            _ => None,
        };

        let label = if let Token::Colon = self.lex.peek()? {
            self.lex.advance()?;
            let name = self.expect_ident()?;
            Some(self.label_catalog.get_or_register(&name))
        } else {
            None
        };

        let predicates = if let Token::LBrace = self.lex.peek()? {
            self.parse_inline_props(alias.as_deref())?
        } else {
            Vec::new()
        };

        self.expect_rparen()?;
        Ok((alias, label, predicates))
    }

    /// Parses `r:TYPE*min..max]` (opening `[` already consumed).
    fn parse_rel_spec(
        &mut self,
    ) -> Result<(Option<String>, Option<u16>, u8, u8), ParseError> {
        let rel_alias = match self.lex.peek()? {
            Token::Ident(_) => Some(self.expect_ident()?),
            _ => None,
        };

        let rel_type = if let Token::Colon = self.lex.peek()? {
            self.lex.advance()?;
            let name = self.expect_ident()?;
            // get_or_register returns Option<u16>; treat catalog overflow as None.
            self.rel_catalog.get_or_register(&name)
        } else {
            None
        };

        let (min_hops, max_hops) = if let Token::Star = self.lex.peek()? {
            self.lex.advance()?; // consume `*`
            match self.lex.peek()? {
                Token::IntLit(_) => {
                    let min = self.expect_uint()? as u8;
                    if let Token::DotDot = self.lex.peek()? {
                        self.lex.advance()?; // consume `..`
                        let max = self.expect_uint()? as u8;
                        (min, max)
                    } else {
                        (min, min)
                    }
                }
                _ => (1, 255), // `*` alone = unlimited hops (bounded by executor)
            }
        } else {
            (1, 1)
        };

        // Consume `]`
        match self.lex.advance()? {
            Token::RBracket => {}
            _ => return Err(self.err("Expected `]` to close relationship spec")),
        }

        Ok((rel_alias, rel_type, min_hops, max_hops))
    }

    /// Parses `{ key: value, ... }` inline property filter as scalar predicates.
    fn parse_inline_props(
        &mut self,
        alias: Option<&str>,
    ) -> Result<Vec<ScalarPredicate>, ParseError> {
        self.lex.advance()?; // consume `{`
        let mut preds = Vec::new();
        loop {
            if let Token::RBrace = self.lex.peek()? {
                self.lex.advance()?;
                break;
            }
            let key = self.expect_ident()?;
            match self.lex.advance()? {
                Token::Colon | Token::Eq => {}
                _ => return Err(self.err("Expected `:` or `=` in property filter")),
            }
            let val = self.parse_literal_value()?;
            let alias_str = alias.unwrap_or("_").to_string();
            preds.push(ScalarPredicate::Eq(
                PredicateValue::PropertyRef { alias: alias_str, key },
                PredicateValue::Literal(val),
            ));
            if let Token::Comma = self.lex.peek()? {
                self.lex.advance()?;
            }
        }
        Ok(preds)
    }

    // ── WHERE predicate ───────────────────────────────────────────────────

    fn parse_predicate(&mut self) -> Result<ScalarPredicate, ParseError> {
        self.parse_or_pred()
    }

    fn parse_or_pred(&mut self) -> Result<ScalarPredicate, ParseError> {
        let mut left = self.parse_and_pred()?;
        while let Token::Or = self.lex.peek()? {
            self.lex.advance()?;
            let right = self.parse_and_pred()?;
            left = ScalarPredicate::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and_pred(&mut self) -> Result<ScalarPredicate, ParseError> {
        let mut left = self.parse_not_pred()?;
        while let Token::And = self.lex.peek()? {
            self.lex.advance()?;
            let right = self.parse_not_pred()?;
            left = ScalarPredicate::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not_pred(&mut self) -> Result<ScalarPredicate, ParseError> {
        if let Token::Not = self.lex.peek()? {
            self.lex.advance()?;
            let inner = self.parse_not_pred()?;
            return Ok(ScalarPredicate::Not(Box::new(inner)));
        }
        self.parse_atom_pred()
    }

    fn parse_atom_pred(&mut self) -> Result<ScalarPredicate, ParseError> {
        // Parenthesised predicate.
        if let Token::LParen = self.lex.peek()? {
            self.lex.advance()?;
            let inner = self.parse_predicate()?;
            self.expect_rparen()?;
            return Ok(inner);
        }

        // PropertyRef: `alias.key`  or bare ident (parameter).
        let lhs = self.parse_pred_value()?;

        match self.lex.peek()? {
            Token::Is => {
                self.lex.advance()?;
                if let Token::Not = self.lex.peek()? {
                    self.lex.advance()?;
                    match self.lex.advance()? {
                        Token::Null => Ok(ScalarPredicate::IsNotNull(lhs)),
                        _ => Err(self.err("Expected NULL after IS NOT")),
                    }
                } else {
                    match self.lex.advance()? {
                        Token::Null => Ok(ScalarPredicate::IsNull(lhs)),
                        _ => Err(self.err("Expected NULL after IS")),
                    }
                }
            }
            Token::Eq => {
                self.lex.advance()?;
                Ok(ScalarPredicate::Eq(lhs, self.parse_pred_value()?))
            }
            Token::Ne => {
                self.lex.advance()?;
                Ok(ScalarPredicate::Ne(lhs, self.parse_pred_value()?))
            }
            Token::Lt => {
                self.lex.advance()?;
                Ok(ScalarPredicate::Lt(lhs, self.parse_pred_value()?))
            }
            Token::Le => {
                self.lex.advance()?;
                Ok(ScalarPredicate::Le(lhs, self.parse_pred_value()?))
            }
            Token::Gt => {
                self.lex.advance()?;
                Ok(ScalarPredicate::Gt(lhs, self.parse_pred_value()?))
            }
            Token::Ge => {
                self.lex.advance()?;
                Ok(ScalarPredicate::Ge(lhs, self.parse_pred_value()?))
            }
            _ => Err(self.err("Expected comparison operator in predicate")),
        }
    }

    fn parse_pred_value(&mut self) -> Result<PredicateValue, ParseError> {
        match self.lex.peek()? {
            Token::Dollar => {
                self.lex.advance()?;
                let name = self.expect_ident()?;
                Ok(PredicateValue::Parameter(name))
            }
            Token::StringLit(_) | Token::IntLit(_) | Token::FloatLit(_)
            | Token::True | Token::False | Token::Null => {
                let val = self.parse_literal_value()?;
                Ok(PredicateValue::Literal(val))
            }
            Token::Ident(_) => {
                let alias = self.expect_ident()?;
                if let Token::Dot = self.lex.peek()? {
                    self.lex.advance()?;
                    let key = self.expect_ident()?;
                    Ok(PredicateValue::PropertyRef { alias, key })
                } else {
                    // bare ident treated as parameter reference
                    Ok(PredicateValue::Parameter(alias))
                }
            }
            _ => Err(self.err("Expected predicate operand")),
        }
    }

    fn parse_literal_value(&mut self) -> Result<serde_json::Value, ParseError> {
        match self.lex.advance()? {
            Token::StringLit(s) => Ok(serde_json::Value::String(s.to_string())),
            Token::IntLit(s) => {
                let n: i64 = s.parse().map_err(|_| self.err("Invalid integer literal"))?;
                Ok(serde_json::Value::Number(n.into()))
            }
            Token::FloatLit(s) => {
                let f: f64 = s.parse().map_err(|_| self.err("Invalid float literal"))?;
                Ok(serde_json::Number::from_f64(f)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null))
            }
            Token::True => Ok(serde_json::Value::Bool(true)),
            Token::False => Ok(serde_json::Value::Bool(false)),
            Token::Null => Ok(serde_json::Value::Null),
            _ => Err(self.err("Expected literal value")),
        }
    }

    // ── RETURN clause ─────────────────────────────────────────────────────

    fn parse_return_clause(&mut self) -> Result<ReturnClause, ParseError> {
        let mut items = Vec::new();

        loop {
            match self.lex.peek()? {
                Token::Limit | Token::Eof | Token::Where | Token::Match => break,
                _ => {}
            }

            let alias = self.expect_ident()?;
            let item = if let Token::Dot = self.lex.peek()? {
                self.lex.advance()?;
                let key = self.expect_ident()?;
                ReturnItem::PropertyRef { alias, key }
            } else {
                ReturnItem::Alias(alias)
            };
            items.push(item);

            if let Token::Comma = self.lex.peek()? {
                self.lex.advance()?;
            } else {
                break;
            }
        }

        let limit = if let Token::Limit = self.lex.peek()? {
            self.lex.advance()?;
            Some(self.expect_uint()?)
        } else {
            None
        };

        Ok(ReturnClause { items, limit })
    }

    // ── CREATE / DELETE mutations ─────────────────────────────────────────

    fn parse_create_clause(&mut self) -> Result<Vec<GraphMutationClause>, ParseError> {
        let mut mutations = Vec::new();
        loop {
            if let Token::LParen = self.lex.peek()? {
                self.lex.advance()?; // consume `(`
                let alias = self.expect_ident()?;
                let labels = if let Token::Colon = self.lex.peek()? {
                    self.lex.advance()?;
                    let name = self.expect_ident()?;
                    vec![self.label_catalog.get_or_register(&name)]
                } else {
                    Vec::new()
                };
                let props = if let Token::LBrace = self.lex.peek()? {
                    self.parse_json_props()?
                } else {
                    std::collections::HashMap::new()
                };
                self.expect_rparen()?;
                mutations.push(GraphMutationClause::CreateNode { alias, labels, properties: props });
            } else {
                break;
            }
            if let Token::Comma = self.lex.peek()? {
                self.lex.advance()?;
            } else {
                break;
            }
        }
        Ok(mutations)
    }

    fn parse_delete_clause(&mut self) -> Result<Vec<GraphMutationClause>, ParseError> {
        let mut mutations = Vec::new();
        loop {
            let alias = self.expect_ident()?;
            mutations.push(GraphMutationClause::DeleteAlias(alias));
            if let Token::Comma = self.lex.peek()? {
                self.lex.advance()?;
            } else {
                break;
            }
        }
        Ok(mutations)
    }

    /// Parses `{ "key": value, ... }` as a JSON-style property map.
    fn parse_json_props(
        &mut self,
    ) -> Result<std::collections::HashMap<String, serde_json::Value>, ParseError> {
        self.lex.advance()?; // consume `{`
        let mut map = std::collections::HashMap::new();
        loop {
            if let Token::RBrace = self.lex.peek()? {
                self.lex.advance()?;
                break;
            }
            let key = self.expect_ident()?;
            match self.lex.advance()? {
                Token::Colon | Token::Eq => {}
                _ => return Err(self.err("Expected `:` in property map")),
            }
            let val = self.parse_literal_value()?;
            map.insert(key, val);
            if let Token::Comma = self.lex.peek()? {
                self.lex.advance()?;
            }
        }
        Ok(map)
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    fn expect_lparen(&mut self) -> Result<(), ParseError> {
        match self.lex.advance()? {
            Token::LParen => Ok(()),
            _ => Err(self.err("Expected `(`")),
        }
    }

    fn expect_rparen(&mut self) -> Result<(), ParseError> {
        match self.lex.advance()? {
            Token::RParen => Ok(()),
            _ => Err(self.err("Expected `)`")),
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.lex.advance()? {
            // Any keyword token that is also a valid identifier in a property-reference
            // context (GraphQuery allows `n.where`, `r.return`, etc.).
            Token::Ident(s) => Ok(s.to_string()),
            // Allow keyword tokens to act as identifiers where unambiguous.
            Token::Match     => Ok("match".to_string()),
            Token::Where     => Ok("where".to_string()),
            Token::Return    => Ok("return".to_string()),
            Token::With      => Ok("with".to_string()),
            Token::As        => Ok("as".to_string()),
            Token::Set       => Ok("set".to_string()),
            Token::Create    => Ok("create".to_string()),
            Token::Delete    => Ok("delete".to_string()),
            Token::Merge     => Ok("merge".to_string()),
            Token::True      => Ok("true".to_string()),
            Token::False     => Ok("false".to_string()),
            Token::Null      => Ok("null".to_string()),
            _ => Err(self.err("Expected identifier")),
        }
    }

    fn expect_uint(&mut self) -> Result<usize, ParseError> {
        match self.lex.advance()? {
            Token::IntLit(s) => s.parse::<usize>().map_err(|_| self.err("Expected unsigned integer")),
            _ => Err(self.err("Expected integer literal")),
        }
    }

    #[cold]
    fn err(&self, msg: &str) -> ParseError {
        ParseError::Invalid(msg.to_string())
    }
}

// ── Public entry-point ────────────────────────────────────────────────────────

/// Parses a GraphQuery query string into a [`QueryAst`].
///
/// `label_catalog` and `rel_catalog` are used to resolve label and relationship-type
/// names to their compact integer IDs at parse time — no second pass required.
///
/// ## Zero-copy guarantee
/// The lexer operates over the original `src` bytes.  Only the final `String`
/// fields in `QueryAst` (one per unique identifier) perform heap allocation.
pub fn parse_query(
    src: &str,
    label_catalog: &LabelCatalog,
    rel_catalog: &RelTypeCatalog,
) -> Result<QueryAst, ParseError> {
    Parser::new(src, label_catalog, rel_catalog).parse()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::catalog::labels::LabelCatalog;
    use crate::graph::catalog::relationships::RelTypeCatalog;
    use crate::graph::query::ast::{Direction, GraphPattern, ReturnItem};
    use crate::graph::query::planner::QueryPlanner;

    fn catalogs() -> (LabelCatalog, RelTypeCatalog) {
        let lc = LabelCatalog::default();
        let rc = RelTypeCatalog::default();
        // Pre-register the types used in tests so IDs are deterministic.
        lc.get_or_register("Person");
        lc.get_or_register("Company");
        lc.get_or_register("VC");
        rc.get_or_register("WORKS_AT");
        rc.get_or_register("INVESTED_IN");
        (lc, rc)
    }

    // ── Lexer round-trips ─────────────────────────────────────────────────

    #[test]
    fn test_lex_basic_tokens() {
        use crate::graph::query::lexer::{Lexer, Token};
        let mut lex = Lexer::new("MATCH (n:Person) RETURN n");
        assert_eq!(lex.advance().unwrap(), Token::Match);
        assert_eq!(lex.advance().unwrap(), Token::LParen);
        assert!(matches!(lex.advance().unwrap(), Token::Ident("n")));
        assert_eq!(lex.advance().unwrap(), Token::Colon);
        assert!(matches!(lex.advance().unwrap(), Token::Ident("Person")));
        assert_eq!(lex.advance().unwrap(), Token::RParen);
        assert_eq!(lex.advance().unwrap(), Token::Return);
        assert!(matches!(lex.advance().unwrap(), Token::Ident("n")));
        assert_eq!(lex.advance().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lex_hop_range() {
        use crate::graph::query::lexer::{Lexer, Token};
        let mut lex = Lexer::new("[r:TYPE*1..5]");
        lex.advance().unwrap(); // [
        lex.advance().unwrap(); // r
        lex.advance().unwrap(); // :
        lex.advance().unwrap(); // TYPE
        lex.advance().unwrap(); // *
        assert!(matches!(lex.advance().unwrap(), Token::IntLit("1")));
        assert_eq!(lex.advance().unwrap(), Token::DotDot);
        assert!(matches!(lex.advance().unwrap(), Token::IntLit("5")));
        assert_eq!(lex.advance().unwrap(), Token::RBracket);
    }

    // ── Parser: node-only query ───────────────────────────────────────────

    #[test]
    fn test_parse_node_only() {
        let (lc, rc) = catalogs();
        let ast = parse_query("MATCH (p:Person) RETURN p", &lc, &rc).unwrap();
        assert_eq!(ast.patterns.len(), 1);
        assert!(matches!(
            &ast.patterns[0],
            GraphPattern::NodePattern { alias, .. } if alias == "p"
        ));
        assert_eq!(ast.return_clause.items.len(), 1);
        assert!(matches!(&ast.return_clause.items[0], ReturnItem::Alias(a) if a == "p"));
    }

    // ── Parser: single-hop relationship ──────────────────────────────────

    #[test]
    fn test_parse_single_hop() {
        let (lc, rc) = catalogs();
        let src = "MATCH (p:Person)-[r:WORKS_AT]->(c:Company) RETURN p, c";
        let ast = parse_query(src, &lc, &rc).unwrap();

        let expand = ast.patterns.iter().find(|p| matches!(p, GraphPattern::Expand { .. }));
        assert!(expand.is_some(), "Expected an Expand pattern");

        if let Some(GraphPattern::Expand { direction, min_hops, max_hops, .. }) = expand {
            assert_eq!(*direction, Direction::Outgoing);
            assert_eq!(*min_hops, 1);
            assert_eq!(*max_hops, 1);
        }
    }

    // ── Parser: the target pattern (multi-hop chain, mixed directions) ───

    #[test]
    fn test_parse_multi_hop_chain() {
        let (lc, rc) = catalogs();
        let src = "MATCH (p:Person)-[:WORKS_AT]->(c:Company)<-[:INVESTED_IN]-(v:VC) RETURN p, c, v";
        let ast = parse_query(src, &lc, &rc).unwrap();

        let expands: Vec<_> = ast
            .patterns
            .iter()
            .filter(|p| matches!(p, GraphPattern::Expand { .. }))
            .collect();

        assert_eq!(expands.len(), 2, "Expected 2 Expand steps");

        // First expand: p -> c (outgoing)
        if let GraphPattern::Expand { direction, dst_alias, .. } = expands[0] {
            assert_eq!(*direction, Direction::Outgoing);
            assert_eq!(dst_alias, "c");
        }

        // Second expand: c <- v (incoming on c, so direction is Incoming)
        if let GraphPattern::Expand { direction, dst_alias, .. } = expands[1] {
            assert_eq!(*direction, Direction::Incoming);
            assert_eq!(dst_alias, "v");
        }

        assert_eq!(ast.return_clause.items.len(), 3);
    }

    // ── Parser: variable-length paths ────────────────────────────────────

    #[test]
    fn test_parse_variable_length_path() {
        let (lc, rc) = catalogs();
        let src = "MATCH (a)-[:WORKS_AT*1..3]->(b) RETURN a, b";
        let ast = parse_query(src, &lc, &rc).unwrap();

        let expand = ast.patterns.iter().find(|p| matches!(p, GraphPattern::Expand { .. }));
        assert!(expand.is_some());
        if let Some(GraphPattern::Expand { min_hops, max_hops, .. }) = expand {
            assert_eq!(*min_hops, 1);
            assert_eq!(*max_hops, 3);
        }
    }

    // ── Parser: OPTIONAL MATCH ────────────────────────────────────────────

    #[test]
    fn test_parse_optional_match() {
        let (lc, rc) = catalogs();
        let src = "MATCH (p:Person) OPTIONAL MATCH (p)-[:WORKS_AT]->(c:Company) RETURN p, c";
        let ast = parse_query(src, &lc, &rc).unwrap();

        let optional = ast
            .patterns
            .iter()
            .find(|p| matches!(p, GraphPattern::OptionalExpand { .. }));
        assert!(optional.is_some(), "Expected an OptionalExpand pattern");
    }

    // ── Parser: WHERE clause ─────────────────────────────────────────────

    #[test]
    fn test_parse_where_clause() {
        use crate::graph::query::ast::ScalarPredicate;
        let (lc, rc) = catalogs();
        let src = "MATCH (p:Person) WHERE p.age > 30 RETURN p";
        let ast = parse_query(src, &lc, &rc).unwrap();
        assert!(!ast.where_clause.predicates.is_empty());
        assert!(matches!(ast.where_clause.predicates[0], ScalarPredicate::Gt(..)));
    }

    // ── Parser: LIMIT ─────────────────────────────────────────────────────

    #[test]
    fn test_parse_limit() {
        let (lc, rc) = catalogs();
        let src = "MATCH (n:Person) RETURN n LIMIT 25";
        let ast = parse_query(src, &lc, &rc).unwrap();
        assert_eq!(ast.return_clause.limit, Some(25));
    }

    // ── Parser: VECTOR MATCH extension ────────────────────────────────────

    #[test]
    fn test_parse_vector_match() {
        use crate::graph::query::ast::VectorContract;
        let (lc, rc) = catalogs();
        let src = "VECTOR MATCH (n) USING $q LIMIT 10 CONTRACT CERTIFIED RETURN n";
        let ast = parse_query(src, &lc, &rc).unwrap();
        let vm = ast.vector_match.expect("Expected VECTOR MATCH clause");
        assert_eq!(vm.binding, "n");
        assert_eq!(vm.query_param, "q");
        assert_eq!(vm.k, 10);
        assert_eq!(vm.contract, VectorContract::Certified);
    }

    // ── QueryPlanner: full compile pipeline ──────────────────────────────

    #[test]
    fn test_planner_compiles_full_chain() {
        let (lc, rc) = catalogs();
        let src = "MATCH (p:Person)-[:WORKS_AT]->(c:Company)<-[:INVESTED_IN]-(v:VC) RETURN p, c, v LIMIT 50";
        let compiled = QueryPlanner::compile(src, &lc, &rc, None).unwrap();

        // Physical plan must have at least: NodeScan + 2 × Expand + Project + Limit
        assert!(compiled.plan.ops.len() >= 4, "Expected ≥4 physical ops, got {}", compiled.plan.ops.len());

        // Column names should be [p, c, v]
        assert_eq!(compiled.column_names, vec!["p", "c", "v"]);
    }

    // ── QueryPlanner: WHERE + LIMIT ───────────────────────────────────────

    #[test]
    fn test_planner_where_and_limit() {
        let (lc, rc) = catalogs();
        let src = "MATCH (p:Person)-[:WORKS_AT]->(c:Company) WHERE p.name = 'Alice' RETURN p, c LIMIT 5";
        let compiled = QueryPlanner::compile(src, &lc, &rc, None).unwrap();
        assert_eq!(compiled.column_names, vec!["p", "c"]);
        // Should have a Limit op at the tail.
        use crate::graph::query::physical::PhysicalOp;
        assert!(compiled.plan.ops.iter().any(|op| matches!(op, PhysicalOp::Limit { .. })));
    }
}
