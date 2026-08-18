/* hnsqr/src/vector/mod.rs */
//!▫~•◦-------------------------------‣
//! # Vector Representation, Folding & Quantization Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod folding;
pub mod quantization;

pub use folding::{ComplexWeaver, GatewayRouter, create_http_router, run_http_server};
pub use quantization::PolarQuantizedVector;
