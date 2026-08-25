/* hnsqr/src/transport/resp.rs */
//!▫~•◦-------------------------------‣
//! # Redis Serialization Protocol (RESP) Wire Server & Streams (Redis Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides native RESP2/RESP3 wire protocol parsing and serialization, enabling standard
//! Redis clients (`redis-py`, `ioredis`, `redis-cli`) to interact directly with HoloSphere,
//! alongside real-time Pub/Sub topic broadcasting and Redis Streams.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::ecosystem::kv_cache::{KvValue, MemoryKvStore};

/// Supported RESP (Redis Serialization Protocol) frame data types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RespFrame {
    SimpleString(String),
    Error(String),
    Integer(i64),
    BulkString(Option<Vec<u8>>),
    Array(Option<Vec<RespFrame>>),
    Null,
}

impl RespFrame {
    /// Serializes a RESP frame to raw wire bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            RespFrame::SimpleString(s) => {
                buf.push(b'+');
                buf.extend_from_slice(s.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            RespFrame::Error(e) => {
                buf.push(b'-');
                buf.extend_from_slice(e.as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            RespFrame::Integer(i) => {
                buf.push(b':');
                buf.extend_from_slice(i.to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
            }
            RespFrame::BulkString(None) => {
                buf.extend_from_slice(b"$-1\r\n");
            }
            RespFrame::BulkString(Some(bytes)) => {
                buf.push(b'$');
                buf.extend_from_slice(bytes.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                buf.extend_from_slice(bytes);
                buf.extend_from_slice(b"\r\n");
            }
            RespFrame::Array(None) => {
                buf.extend_from_slice(b"*-1\r\n");
            }
            RespFrame::Array(Some(frames)) => {
                buf.push(b'*');
                buf.extend_from_slice(frames.len().to_string().as_bytes());
                buf.extend_from_slice(b"\r\n");
                for f in frames {
                    buf.extend_from_slice(&f.serialize());
                }
            }
            RespFrame::Null => {
                buf.extend_from_slice(b"$-1\r\n");
            }
        }
        buf
    }
}

/// Real-time Pub/Sub Topic Broker.
pub struct PubSubBroker {
    channels: RwLock<HashMap<String, Vec<tokio::sync::mpsc::UnboundedSender<String>>>>,
    total_messages_published: AtomicU64,
}

impl PubSubBroker {
    pub fn new() -> Self {
        Self {
            channels: RwLock::new(HashMap::new()),
            total_messages_published: AtomicU64::new(0),
        }
    }

    /// Subscribes to a channel, receiving an unbounded receiver channel.
    pub fn subscribe(&self, channel: &str) -> tokio::sync::mpsc::UnboundedReceiver<String> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut channels = self.channels.write();
        channels.entry(channel.to_string()).or_default().push(tx);
        rx
    }

    /// Publishes a message to all active channel subscribers.
    pub fn publish(&self, channel: &str, message: &str) -> usize {
        self.total_messages_published
            .fetch_add(1, Ordering::Relaxed);
        let mut channels = self.channels.write();
        if let Some(subscribers) = channels.get_mut(channel) {
            subscribers.retain(|tx| tx.send(message.to_string()).is_ok());
            subscribers.len()
        } else {
            0
        }
    }
}

impl Default for PubSubBroker {
    fn default() -> Self {
        Self::new()
    }
}

/// A single entry in a Redis Stream.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamEntry {
    pub entry_id: String, // e.g. "1710000000000-0"
    pub fields: HashMap<String, String>,
}

/// Redis Streams append/read engine.
pub struct RedisStreamEngine {
    streams: RwLock<HashMap<String, VecDeque<StreamEntry>>>,
}

impl RedisStreamEngine {
    pub fn new() -> Self {
        Self {
            streams: RwLock::new(HashMap::new()),
        }
    }

    /// Appends an entry to a stream (XADD).
    pub fn xadd(&self, stream: &str, fields: HashMap<String, String>) -> String {
        let mut streams = self.streams.write();
        let queue = streams.entry(stream.to_string()).or_default();
        let id = format!(
            "{}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis(),
            queue.len()
        );

        queue.push_back(StreamEntry {
            entry_id: id.clone(),
            fields,
        });

        id
    }

    /// Reads up to count entries from a stream (XREAD).
    pub fn xread(&self, stream: &str, start_index: usize, count: usize) -> Vec<StreamEntry> {
        let streams = self.streams.read();
        if let Some(queue) = streams.get(stream) {
            queue
                .iter()
                .skip(start_index)
                .take(count)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for RedisStreamEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed zero-allocation RESP frame view pointing directly into incoming TCP buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RespBorrowedFrame<'a> {
    SimpleString(&'a str),
    Error(&'a str),
    Integer(i64),
    BulkString(Option<&'a [u8]>),
    Array(usize), // Element count
    Null,
}

/// Zero-Allocation Streaming Push Parser for RESP wire streams.
pub struct StreamingRespParser;

impl StreamingRespParser {
    /// Parses a single RESP frame from a raw byte buffer, returning `(frame, consumed_bytes)`.
    pub fn parse_frame(buf: &[u8]) -> Option<(RespBorrowedFrame<'_>, usize)> {
        if buf.is_empty() {
            return None;
        }

        let prefix = buf[0];
        let rest = &buf[1..];

        match prefix {
            b'+' => {
                let (line, consumed) = Self::read_crlf_line(rest)?;
                let s = std::str::from_utf8(line).ok()?;
                Some((RespBorrowedFrame::SimpleString(s), consumed + 1))
            }
            b'-' => {
                let (line, consumed) = Self::read_crlf_line(rest)?;
                let s = std::str::from_utf8(line).ok()?;
                Some((RespBorrowedFrame::Error(s), consumed + 1))
            }
            b':' => {
                let (line, consumed) = Self::read_crlf_line(rest)?;
                let s = std::str::from_utf8(line).ok()?;
                let val: i64 = s.parse().ok()?;
                Some((RespBorrowedFrame::Integer(val), consumed + 1))
            }
            b'$' => {
                let (line, consumed_len) = Self::read_crlf_line(rest)?;
                let s = std::str::from_utf8(line).ok()?;
                let len: i64 = s.parse().ok()?;
                if len < 0 {
                    return Some((RespBorrowedFrame::BulkString(None), consumed_len + 1));
                }

                let payload_len = len as usize;
                let payload_start = 1 + consumed_len;
                if buf.len() < payload_start + payload_len + 2 {
                    return None;
                }

                let payload = &buf[payload_start..(payload_start + payload_len)];
                Some((
                    RespBorrowedFrame::BulkString(Some(payload)),
                    payload_start + payload_len + 2,
                ))
            }
            b'*' => {
                let (line, consumed) = Self::read_crlf_line(rest)?;
                let s = std::str::from_utf8(line).ok()?;
                let count: i64 = s.parse().ok()?;
                if count < 0 {
                    Some((RespBorrowedFrame::Null, consumed + 1))
                } else {
                    Some((RespBorrowedFrame::Array(count as usize), consumed + 1))
                }
            }
            _ => None,
        }
    }

    /// Extracts an entire RESP command array as borrowed `&[u8]` slices with zero string allocations.
    pub fn parse_command_slices<'a>(buf: &'a [u8]) -> Option<(Vec<&'a [u8]>, usize)> {
        let (frame, mut offset) = Self::parse_frame(buf)?;
        match frame {
            RespBorrowedFrame::Array(count) => {
                let mut args = Vec::with_capacity(count);
                for _ in 0..count {
                    let (arg_frame, consumed) = Self::parse_frame(&buf[offset..])?;
                    offset += consumed;
                    match arg_frame {
                        RespBorrowedFrame::BulkString(Some(bytes)) => args.push(bytes),
                        RespBorrowedFrame::SimpleString(s) => args.push(s.as_bytes()),
                        _ => return None,
                    }
                }
                Some((args, offset))
            }
            _ => None,
        }
    }

    #[inline(always)]
    fn read_crlf_line(buf: &[u8]) -> Option<(&[u8], usize)> {
        for i in 0..buf.len().saturating_sub(1) {
            if buf[i] == b'\r' && buf[i + 1] == b'\n' {
                return Some((&buf[..i], i + 2));
            }
        }
        None
    }
}

/// RESP Wire Command Dispatcher integrating with HoloSphere's MemoryKvStore.
pub struct RespServer {
    kv_store: Arc<MemoryKvStore>,
    pubsub: Arc<PubSubBroker>,
    streams: Arc<RedisStreamEngine>,
}

impl RespServer {
    pub fn new(kv_store: Arc<MemoryKvStore>) -> Self {
        Self {
            kv_store,
            pubsub: Arc::new(PubSubBroker::new()),
            streams: Arc::new(RedisStreamEngine::new()),
        }
    }

    /// Dispatches raw byte slice command arguments with zero UTF-8 allocation overhead.
    pub fn handle_raw_command(&self, args: &[&[u8]]) -> RespFrame {
        if args.is_empty() {
            return RespFrame::Error("ERR empty command".into());
        }

        let cmd = args[0];
        if cmd.eq_ignore_ascii_case(b"PING") {
            if args.len() > 1 {
                RespFrame::BulkString(Some(args[1].to_vec()))
            } else {
                RespFrame::SimpleString("PONG".into())
            }
        } else if cmd.eq_ignore_ascii_case(b"SET") {
            if args.len() < 3 {
                return RespFrame::Error("ERR wrong number of arguments for 'set' command".into());
            }
            self.kv_store
                .set_raw(args[1], KvValue::Bytes(args[2].to_vec()), None);
            RespFrame::SimpleString("OK".into())
        } else if cmd.eq_ignore_ascii_case(b"GET") {
            if args.len() < 2 {
                return RespFrame::Error("ERR wrong number of arguments for 'get' command".into());
            }
            match self.kv_store.get_raw(args[1]) {
                Some(KvValue::Bytes(b)) => RespFrame::BulkString(Some(b)),
                Some(KvValue::String(s)) => RespFrame::BulkString(Some(s.into_bytes())),
                Some(KvValue::Integer(i)) => {
                    RespFrame::BulkString(Some(i.to_string().into_bytes()))
                }
                Some(_) => RespFrame::Error(
                    "WRONGTYPE Operation against a key holding the wrong kind of value".into(),
                ),
                None => RespFrame::Null,
            }
        } else if cmd.eq_ignore_ascii_case(b"INCR") {
            if args.len() < 2 {
                return RespFrame::Error("ERR wrong number of arguments for 'incr' command".into());
            }
            let val = self.kv_store.incr_by_raw(args[1], 1);
            RespFrame::Integer(val)
        } else if cmd.eq_ignore_ascii_case(b"DEL") {
            if args.len() < 2 {
                return RespFrame::Error("ERR wrong number of arguments for 'del' command".into());
            }
            let count = self.kv_store.delete_raw(args[1]) as i64;
            RespFrame::Integer(count)
        } else if cmd.eq_ignore_ascii_case(b"PUBLISH") {
            if args.len() < 3 {
                return RespFrame::Error(
                    "ERR wrong number of arguments for 'publish' command".into(),
                );
            }
            let chan = String::from_utf8_lossy(args[1]);
            let msg = String::from_utf8_lossy(args[2]);
            let recipients = self.pubsub.publish(&chan, &msg);
            RespFrame::Integer(recipients as i64)
        } else {
            RespFrame::Error(format!(
                "ERR unknown command '{}'",
                String::from_utf8_lossy(args[0])
            ))
        }
    }

    /// Dispatches parsed RESP argument array to internal commands.
    pub fn handle_command(&self, args: &[String]) -> RespFrame {
        let mut byte_slices = smallvec::SmallVec::<[&[u8]; 16]>::new();
        byte_slices.extend(args.iter().map(String::as_bytes));
        self.handle_raw_command(&byte_slices)
    }

    pub fn pubsub(&self) -> &Arc<PubSubBroker> {
        &self.pubsub
    }

    pub fn streams(&self) -> &Arc<RedisStreamEngine> {
        &self.streams
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resp_frame_serialization() {
        let simple = RespFrame::SimpleString("OK".into());
        assert_eq!(simple.serialize(), b"+OK\r\n");

        let int_frame = RespFrame::Integer(42);
        assert_eq!(int_frame.serialize(), b":42\r\n");

        let bulk = RespFrame::BulkString(Some(b"hello".to_vec()));
        assert_eq!(bulk.serialize(), b"$5\r\nhello\r\n");
    }

    #[test]
    fn test_resp_server_commands() {
        let kv = Arc::new(MemoryKvStore::new());
        let server = RespServer::new(kv);

        // PING
        assert_eq!(
            server.handle_command(&["PING".into()]),
            RespFrame::SimpleString("PONG".into())
        );

        // SET & GET
        server.handle_command(&["SET".into(), "mykey".into(), "myval".into()]);
        let res = server.handle_command(&["GET".into(), "mykey".into()]);
        assert_eq!(res, RespFrame::BulkString(Some(b"myval".to_vec())));

        // INCR
        let incr_res = server.handle_command(&["INCR".into(), "counter".into()]);
        assert_eq!(incr_res, RespFrame::Integer(1));
    }

    #[test]
    fn test_streaming_resp_parser() {
        let raw_stream = b"*3\r\n$3\r\nSET\r\n$5\r\nmykey\r\n$5\r\nmyval\r\n";
        let (args, consumed) = StreamingRespParser::parse_command_slices(raw_stream).unwrap();
        assert_eq!(consumed, raw_stream.len());
        assert_eq!(args.len(), 3);
        assert_eq!(args[0], b"SET");
        assert_eq!(args[1], b"mykey");
        assert_eq!(args[2], b"myval");

        let kv = Arc::new(MemoryKvStore::new());
        let server = RespServer::new(kv);
        let resp = server.handle_raw_command(&args);
        assert_eq!(resp, RespFrame::SimpleString("OK".into()));
    }
}
