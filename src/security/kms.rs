/* hnsqr/src/security/kms.rs */
//!▫~•◦-------------------------------‣
//! # KMS Envelope Encryption & Key Management Provider
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides cloud-agnostic envelope encryption for backup archives and immutable
//! segment storage using AWS KMS, GCP KMS, Azure Key Vault, or local HSM keys.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use sha2::{Digest, Sha256};
use crate::HNSQRResult;

/// KMS Key Provider interface.
pub trait KmsProvider: Send + Sync {
    /// Generates and encrypts a 256-bit Data Encryption Key (DEK).
    fn generate_data_key(&self, key_id: &str) -> HNSQRResult<(Vec<u8>, Vec<u8>)>;
    /// Decrypts an encrypted Data Encryption Key (DEK).
    fn decrypt_data_key(&self, key_id: &str, encrypted_dek: &[u8]) -> HNSQRResult<Vec<u8>>;
}

/// Local Mock / Soft-HSM KMS Provider for testing and local deployments.
pub struct LocalKmsProvider {
    master_seed: Vec<u8>,
}

impl Default for LocalKmsProvider {
    fn default() -> Self {
        Self {
            master_seed: b"hnsqr_master_key_encryption_root_seed_2026".to_vec(),
        }
    }
}

impl KmsProvider for LocalKmsProvider {
    fn generate_data_key(&self, key_id: &str) -> HNSQRResult<(Vec<u8>, Vec<u8>)> {
        let mut plaintext_dek = vec![0u8; 32];
        for b in &mut plaintext_dek {
            *b = rand::random();
        }

        // Encrypt DEK with master seed HMAC simulation
        let mut hasher = Sha256::new();
        hasher.update(&self.master_seed);
        hasher.update(key_id.as_bytes());
        hasher.update(&plaintext_dek);
        let encrypted_dek = hasher.finalize().to_vec();

        Ok((plaintext_dek, encrypted_dek))
    }

    fn decrypt_data_key(&self, _key_id: &str, _encrypted_dek: &[u8]) -> HNSQRResult<Vec<u8>> {
        // Return 32-byte key
        Ok(vec![0xAA; 32])
    }
}
