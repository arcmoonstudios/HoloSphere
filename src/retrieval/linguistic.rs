/* holosphere/src/retrieval/linguistic.rs */
//!▫~•◦-------------------------------‣
//! # Linguistic Full-Text Engine & Fuzzy Levenshtein Automata (Elasticsearch Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides fuzzy string matching via Levenshtein edit-distance automata ($\le k$ typo tolerance),
//! algorithmic Porter morphological stemming, phonetic Soundex encoding, and CJK n-gram segmentation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashSet;

/// Supported linguistic analyzer modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageMode {
    English,
    German,
    Spanish,
    CjkNgram,
}

/// Fuzzy Levenshtein Automaton evaluating edit distance against search candidate terms.
pub struct FuzzyLevenshteinAutomaton {
    query_chars: Vec<char>,
    max_edit_distance: usize,
}

impl FuzzyLevenshteinAutomaton {
    pub fn new(query: &str, max_edits: usize) -> Self {
        Self {
            query_chars: query.to_lowercase().chars().collect(),
            max_edit_distance: max_edits,
        }
    }

    /// Evaluates candidate string edit distance using Wagner-Fischer dynamic programming.
    pub fn matches(&self, candidate: &str) -> (bool, usize) {
        let cand_chars: Vec<char> = candidate.to_lowercase().chars().collect();
        let m = self.query_chars.len();
        let n = cand_chars.len();

        if (m as isize - n as isize).unsigned_abs() > self.max_edit_distance {
            return (false, usize::MAX);
        }

        // DERIVED: Replaces O(M * N) 2D matrix allocations (m + 1 heap vectors) with 2 row buffer vectors.
        let mut prev = (0..=n).collect::<Vec<usize>>();
        let mut curr = vec![0usize; n + 1];

        for i in 1..=m {
            curr[0] = i;
            for j in 1..=n {
                let cost = if self.query_chars[i - 1] == cand_chars[j - 1] {
                    0
                } else {
                    1
                };
                curr[j] = (prev[j] + 1) // deletion
                    .min(curr[j - 1] + 1) // insertion
                    .min(prev[j - 1] + cost); // substitution
            }
            std::mem::swap(&mut prev, &mut curr);
        }

        let dist = prev[n];
        (dist <= self.max_edit_distance, dist)
    }
}

/// Algorithmic Morphological Stemmer and Tokenizer.
pub struct MorphologicalStemmer {
    stopwords: HashSet<String>,
}

impl MorphologicalStemmer {
    pub fn new() -> Self {
        let mut stopwords = HashSet::new();
        for word in &[
            "the", "is", "are", "at", "which", "on", "and", "a", "an", "in", "to", "for", "of",
            "with",
        ] {
            stopwords.insert(word.to_string());
        }
        Self { stopwords }
    }

    /// Tokenizes and stems a natural language string into root terms.
    pub fn tokenize_and_stem(&self, text: &str, mode: LanguageMode) -> Vec<String> {
        if mode == LanguageMode::CjkNgram {
            // Bigram segmentation for CJK characters
            let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
            if chars.len() <= 1 {
                return chars.into_iter().map(|c| c.to_string()).collect();
            }
            let mut bigrams = Vec::new();
            for w in chars.windows(2) {
                bigrams.push(format!("{}{}", w[0], w[1]));
            }
            return bigrams;
        }

        let mut tokens = Vec::new();
        for raw in text.split(|c: char| !c.is_alphanumeric()) {
            let lower = raw.to_lowercase();
            if lower.is_empty() || self.stopwords.contains(&lower) {
                continue;
            }
            tokens.push(self.stem_word(&lower));
        }
        tokens
    }

    /// Step 1 algorithmic suffix stripping (simplified Porter stemmer).
    fn stem_word(&self, word: &str) -> String {
        let mut s = word.to_string();
        if s.ends_with("sses") {
            s.truncate(s.len() - 2);
        } else if s.ends_with("ies") {
            s.truncate(s.len() - 2);
        } else if s.ends_with("ing") && s.len() > 5 {
            s.truncate(s.len() - 3);
        } else if s.ends_with("ed") && s.len() > 4 {
            s.truncate(s.len() - 2);
        } else if s.ends_with('s') && !s.ends_with("ss") && s.len() > 3 {
            s.pop();
        }
        s
    }
}

impl Default for MorphologicalStemmer {
    fn default() -> Self {
        Self::new()
    }
}

/// Phonetic Soundex encoder mapping sounds to 4-character phonetic codes.
pub struct PhoneticMatcher;

impl PhoneticMatcher {
    /// Computes American Soundex representation (e.g. "Robert" -> "R163").
    pub fn soundex(word: &str) -> String {
        let clean: Vec<char> = word
            .to_uppercase()
            .chars()
            .filter(|c| c.is_alphabetic())
            .collect();
        if clean.is_empty() {
            return String::new();
        }

        let first = clean[0];
        let mut code = String::with_capacity(4);
        code.push(first);

        let mut prev_digit = Self::map_char_to_digit(first);

        for &c in &clean[1..] {
            let digit = Self::map_char_to_digit(c);
            if digit != '0' && digit != prev_digit {
                code.push(digit);
                if code.len() == 4 {
                    break;
                }
            }
            prev_digit = digit;
        }

        while code.len() < 4 {
            code.push('0');
        }

        code
    }

    fn map_char_to_digit(c: char) -> char {
        match c {
            'B' | 'F' | 'P' | 'V' => '1',
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
            'D' | 'T' => '3',
            'L' => '4',
            'M' | 'N' => '5',
            'R' => '6',
            _ => '0',
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_levenshtein_automaton() {
        let dfa = FuzzyLevenshteinAutomaton::new("holosphere", 2);
        assert!(dfa.matches("holosphere").0); // exact (0 edits)
        assert!(dfa.matches("holosfere").0); // 2 edits (ph -> f)
        assert!(dfa.matches("holospher").0); // 1 edit (trailing e dropped)
        assert!(!dfa.matches("completely_other").0);
    }

    #[test]
    fn test_morphological_stemmer_and_cjk() {
        let stemmer = MorphologicalStemmer::new();
        let tokens =
            stemmer.tokenize_and_stem("The engineering teams are working", LanguageMode::English);
        assert_eq!(tokens, vec!["engineer", "team", "work"]);

        let cjk = stemmer.tokenize_and_stem("量子計算", LanguageMode::CjkNgram);
        assert_eq!(cjk, vec!["量子", "子計", "計算"]);
    }

    #[test]
    fn test_phonetic_soundex() {
        assert_eq!(PhoneticMatcher::soundex("Robert"), "R163");
        assert_eq!(PhoneticMatcher::soundex("Rupert"), "R163");
        assert_eq!(PhoneticMatcher::soundex("HoloSphere"), "H421");
    }
}
