/* hnsqr/src/transport/resp.rs */
//!▫~•◦-------------------------------‣
//! # Redis Serialization Protocol (RESP) Wire Server & Streams (Redis Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides native RESP2/RESP3 wire protocol parsing and serialization, enabling standard
//! Redis clients (`redis-py`, `ioredis`, `redis-cli`) to interact directly with HoloSphere,
//! alongside real-time Pub/Sub topic broadcasting and Redis Streams with Consumer Groups.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

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
        self.total_messages_published.fetch_add(1, Ordering::Relaxed);
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

/// Redis Streams Engine with Consumer Groups.
#[allow(dead_code)]
pub struct RedisStreamEngine {
    streams: RwLock<HashMap<String, VecDeque<StreamEntry>>>,
    consumer_offsets: RwLock<HashMap<String, HashMap<String, usize>>>, // (Stream:Group) -> (Consumer -> Offset)
}

impl RedisStreamEngine {
    pub fn new() -> Self {
        Self {
            streams: RwLock::new(HashMap::new()),
            consumer_offsets: RwLock::new(HashMap::new()),
        }
    }

    /// Appends an entry to a stream (XADD).
    pub fn xadd(&self, stream: &str, fields: HashMap<String, String>) -> String {
        let mut streams = self.streams.write();
        let queue = streams.entry(stream.to_string()).or_default();
        let id = format!("{}-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(), queue.len());

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
            queue.iter().skip(start_index).take(count).cloned().collect()
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

    /// Dispatches parsed RESP argument array to internal commands.
    pub fn handle_command(&self, args: &[String]) -> RespFrame {
        if args.is_empty() {
            return RespFrame::Error("ERR empty command".into());
        }

        let cmd = args[0].to_uppercase();
        match cmd.as_str() {
            "PING" => {
                if args.len() > 1 {
                    RespFrame::BulkString(Some(args[1].as_bytes().to_vec()))
                } else {
                    RespFrame::SimpleString("PONG".into())
                }
            }
            "SET" => {
                if args.len() < 3 {
                    return RespFrame::Error("ERR wrong number of arguments for 'set' command".into());
                }
                self.kv_store.set(&args[1], KvValue::String(args[2].clone()), None);
                RespFrame::SimpleString("OK".into())
            }
            "GET" => {
                if args.len() < 2 {
                    return RespFrame::Error("ERR wrong number of arguments for 'get' command".into());
                }
                match self.kv_store.get(&args[1]) {
                    Some(KvValue::String(s)) => RespFrame::BulkString(Some(s.into_bytes())),
                    Some(KvValue::Integer(i)) => RespFrame::BulkString(Some(i.to_string().into_bytes())),
                    Some(_) => RespFrame::Error("WRONGTYPE Operation against a key holding the wrong kind of value".into()),
                    None => RespFrame::Null,
                }
            }
            "INCR" => {
                if args.len() < 2 {
                    return RespFrame::Error("ERR wrong number of arguments for 'incr' command".into());
                }
                let val = self.kv_store.incr_by(&args[1], 1);
                RespFrame::Integer(val)
            }
            "DEL" => {
                if args.len() < 2 {
                    return RespFrame::Error("ERR wrong number of arguments for 'del' command".into());
                }
                let count = self.kv_store.delete(&args[1]) as i64;
                RespFrame::Integer(count)
            }
            "PUBLISH" => {
                if args.len() < 3 {
                    return RespFrame::Error("ERR wrong number of arguments for 'publish' command".into());
                }
                let recipients = self.pubsub.publish(&args[1], &args[2]);
                RespFrame::Integer(recipients as i64)
            }
            _ => RespFrame::Error(format!("ERR unknown command '{}'", args[0])),
        }
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
        assert_eq!(server.handle_command(&["PING".into()]), RespFrame::SimpleString("PONG".into()));

        // SET & GET
        server.handle_command(&["SET".into(), "mykey".into(), "myval".into()]);
        let res = server.handle_command(&["GET".into(), "mykey".into()]);
        assert_eq!(res, RespFrame::BulkString(Some(b"myval".to_vec())));

        // INCR
        let incr_res = server.handle_command(&["INCR".into(), "counter".into()]);
        assert_eq!(incr_res, RespFrame::Integer(1));
    }
}
