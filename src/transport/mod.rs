/* hnsqr/src/transport/mod.rs */
//!▫~•◦-------------------------------‣
//! # Wire Transport & QIR0 Network Protocol Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod qir0;
pub mod resp;
pub mod web_console;

pub use qir0::{HNSQRClient, HNSQRServer, MessageHeader, OpCode, PROTOCOL_MAGIC};
pub use resp::{PubSubBroker, RedisStreamEngine, RespFrame, RespServer, StreamEntry};
pub use web_console::{CONSOLE_HTML, console_handler};
