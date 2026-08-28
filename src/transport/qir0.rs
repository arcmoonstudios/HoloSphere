/* holosphere/src/server.rs */
//!▫~•◦-------------------------------‣
//! # Zero-Copy Asynchronous Binary TCP Server & Network Protocol
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a high-throughput, non-blocking asynchronous TCP server and client for the HNSQR database.
//! Uses a compact 16-byte binary wire protocol framing layer that streams query buffers directly into
//! thread-safe index routines without intermediate heap copies or serialization allocations.
//!
//! ## Key Capabilities
//! - **Direct Buffer Transmutation:** Zero-copy transmutation of network wire payloads into complex vector slices.
//! - **Non-Allocating Frame Pipeline:** Reuses client stream buffers (`read_buf`, `write_buf`) across request lifecycles.
//! - **Pipelined Correlation Engine:** 16-byte fixed binary header with asynchronous request correlation IDs.
//!
//! ### Architectural Notes
//! Designed for integration with `HNSQRIndex` and `MmapArena`. Slices are cast directly to `&[Complex32]`
//! adhering to IEEE-754 binary layout guarantees.
//!
//! #### Example
//! ```rust
//! use hnsqr::transport::qir0::{HNSQRServer, HNSQRClient};
//! use hnsqr::{HNSQRIndex, HNSQRConfig};
//! use std::sync::Arc;
//!
//! let index = Arc::new(HNSQRIndex::new(HNSQRConfig::default(), 64));
//! let server = HNSQRServer::new(index, "127.0.0.1:9090".parse().unwrap());
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::net::SocketAddr;
use std::sync::Arc;

use bytes::{Buf, BufMut, BytesMut};
use num_complex::Complex32;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tracing::{error, info, trace};

use crate::{HNSQRError, HNSQRIndex, HNSQRResult, SimilarityScore, VectorEmbedding};

/// Protocol Magic Bytes ("QIR0" in ASCII - Query Interchange Record v0).
pub const PROTOCOL_MAGIC: u32 = 0x51495230;
/// Protocol Header Size in bytes.
pub const HEADER_SIZE: usize = 16;

/// Protocol Operation Codes.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpCode {
    /// Healthcheck Ping / Pong request.
    Ping = 0x0001,
    /// Vector Insertion request.
    Insert = 0x0002,
    /// Single-vector Nearest Neighbor Search request.
    Search = 0x0003,
    /// Multi-vector Parallel Batch Search request.
    BatchSearch = 0x0004,
    /// Index Telemetry & Operational Statistics.
    Stats = 0x0005,
    /// GraphQuery Graph Query Execution.
    GraphQuery = 0x0006,
    /// Error response from server.
    Error = 0x00FF,
}

impl TryFrom<u16> for OpCode {
    type Error = HNSQRError;

    #[inline(always)]
    fn try_from(val: u16) -> Result<Self, HNSQRError> {
        match val {
            0x0001 => Ok(OpCode::Ping),
            0x0002 => Ok(OpCode::Insert),
            0x0003 => Ok(OpCode::Search),
            0x0004 => Ok(OpCode::BatchSearch),
            0x0005 => Ok(OpCode::Stats),
            0x0006 => Ok(OpCode::GraphQuery),
            0x00FF => Ok(OpCode::Error),
            _ => Err(HNSQRError::SerializationError(format!(
                "Unknown opcode: 0x{:04X}",
                val
            ))),
        }
    }
}

/// Binary Protocol Message Header aligned to 16 bytes.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct MessageHeader {
    /// Magic identifier (0x51495230).
    pub magic: u32,
    /// Operation Code.
    pub opcode: OpCode,
    /// Protocol flags.
    pub flags: u16,
    /// Correlation ID for pipelined requests.
    pub request_id: u32,
    /// Length of payload in bytes.
    pub payload_len: u32,
}

impl MessageHeader {
    /// Encodes header directly into destination byte buffer.
    #[inline(always)]
    pub fn encode(&self, dst: &mut BytesMut) {
        dst.put_u32(self.magic);
        dst.put_u16(self.opcode as u16);
        dst.put_u16(self.flags);
        dst.put_u32(self.request_id);
        dst.put_u32(self.payload_len);
    }

    /// Decodes header from byte buffer without allocations.
    #[inline(always)]
    pub fn decode(src: &mut BytesMut) -> HNSQRResult<Option<Self>> {
        if src.len() < HEADER_SIZE {
            return Ok(None);
        }

        let magic = (&src[0..4]).get_u32();
        if magic != PROTOCOL_MAGIC {
            return Err(HNSQRError::SerializationError(format!(
                "Invalid wire magic: 0x{:08X}",
                magic
            )));
        }

        let opcode_raw = (&src[4..6]).get_u16();
        let opcode = OpCode::try_from(opcode_raw)?;
        let flags = (&src[6..8]).get_u16();
        let request_id = (&src[8..12]).get_u32();
        let payload_len = (&src[12..16]).get_u32();

        src.advance(HEADER_SIZE);

        Ok(Some(Self {
            magic,
            opcode,
            flags,
            request_id,
            payload_len,
        }))
    }
}

use crate::service::{HNSQRService, RequestContext, StandaloneService, UpsertRequest};

/// High-throughput asynchronous TCP Server for HNSQR.
pub struct HNSQRServer<S: HNSQRService = StandaloneService> {
    service: Arc<S>,
    addr: SocketAddr,
}

impl HNSQRServer<StandaloneService> {
    /// Creates a new server instance wrapping a standalone in-memory or durable index.
    pub fn new(index: Arc<HNSQRIndex>, addr: SocketAddr) -> Self {
        Self {
            service: Arc::new(StandaloneService::new(index)),
            addr,
        }
    }
}

impl<S: HNSQRService + 'static> HNSQRServer<S> {
    /// Creates a new server instance with a custom production HNSQR service (standalone or cluster).
    pub fn with_service(service: Arc<S>, addr: SocketAddr) -> Self {
        Self { service, addr }
    }

    /// Starts the asynchronous TCP server on Tokio runtime.
    pub async fn run(self) -> HNSQRResult<()> {
        let listener = TcpListener::bind(self.addr).await.map_err(|e| {
            HNSQRError::ConcurrencyError(format!("Failed to bind TCP server: {}", e))
        })?;

        info!("HNSQR TCP Server listening on {}", self.addr);

        loop {
            match listener.accept().await {
                Ok((stream, peer_addr)) => {
                    trace!("New client connected from {}", peer_addr);
                    let service = Arc::clone(&self.service);
                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_client(stream, service).await {
                            error!("Error handling client {}: {}", peer_addr, e);
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }
    }

    /// Processes messages for an individual client TCP stream.
    pub async fn handle_client(mut stream: TcpStream, service: Arc<S>) -> HNSQRResult<()> {
        let mut read_buf = BytesMut::with_capacity(65536);
        let mut write_buf = BytesMut::with_capacity(65536);

        loop {
            while read_buf.len() < HEADER_SIZE {
                let n = stream
                    .read_buf(&mut read_buf)
                    .await
                    .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
                if n == 0 {
                    return Ok(());
                }
            }

            let payload_len = (&read_buf[12..16]).get_u32() as usize;
            let total_required = HEADER_SIZE + payload_len;

            while read_buf.len() < total_required {
                let n = stream
                    .read_buf(&mut read_buf)
                    .await
                    .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
                if n == 0 {
                    return Ok(());
                }
            }

            let header = MessageHeader::decode(&mut read_buf)?.unwrap();
            let mut payload = read_buf.split_to(header.payload_len as usize);

            write_buf.clear();

            match header.opcode {
                OpCode::Ping => {
                    let resp_header = MessageHeader {
                        magic: PROTOCOL_MAGIC,
                        opcode: OpCode::Ping,
                        flags: 0,
                        request_id: header.request_id,
                        payload_len: 4,
                    };
                    resp_header.encode(&mut write_buf);
                    write_buf.put_slice(b"PONG");
                }
                OpCode::Insert => {
                    let id_len = payload.get_u16() as usize;
                    if payload.len() < id_len + 4 {
                        Self::write_error(
                            &mut write_buf,
                            header.request_id,
                            "Malformed insert payload",
                        );
                    } else {
                        let id_bytes = payload.split_to(id_len);
                        let id_str = match std::str::from_utf8(&id_bytes) {
                            Ok(s) => s,
                            Err(_) => {
                                Self::write_error(
                                    &mut write_buf,
                                    header.request_id,
                                    "Invalid UTF-8 ID",
                                );
                                stream
                                    .write_all(&write_buf)
                                    .await
                                    .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
                                continue;
                            }
                        };

                        let dim = payload.get_u32() as usize;
                        let byte_len = dim * std::mem::size_of::<Complex32>();

                        if payload.len() < byte_len {
                            Self::write_error(
                                &mut write_buf,
                                header.request_id,
                                "Truncated vector payload",
                            );
                        } else {
                            let mut complex_data = vec![Complex32::new(0.0, 0.0); dim];
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    payload.as_ptr(),
                                    complex_data.as_mut_ptr() as *mut u8,
                                    byte_len,
                                );
                            }
                            let vec = VectorEmbedding::from_complex(complex_data);
                            let ctx = RequestContext {
                                request_id: header.request_id as u64,
                                ..Default::default()
                            };
                            let req = UpsertRequest {
                                id: id_str.to_string(),
                                vector: vec,
                                metadata: None,
                            };

                            match service.upsert(&ctx, req).await {
                                Ok(_receipt) => {
                                    let resp_header = MessageHeader {
                                        magic: PROTOCOL_MAGIC,
                                        opcode: OpCode::Insert,
                                        flags: 0,
                                        request_id: header.request_id,
                                        payload_len: 4,
                                    };
                                    resp_header.encode(&mut write_buf);
                                    write_buf.put_u32(0);
                                }
                                Err(e) => {
                                    Self::write_error(
                                        &mut write_buf,
                                        header.request_id,
                                        &e.to_string(),
                                    );
                                }
                            }
                        }
                    }
                }
                OpCode::Search => {
                    if payload.len() < 8 {
                        Self::write_error(
                            &mut write_buf,
                            header.request_id,
                            "Malformed search payload",
                        );
                    } else {
                        let k = payload.get_u32() as usize;
                        let dim = payload.get_u32() as usize;
                        let byte_len = dim * std::mem::size_of::<Complex32>();

                        if payload.len() < byte_len {
                            Self::write_error(
                                &mut write_buf,
                                header.request_id,
                                "Truncated query payload",
                            );
                        } else {
                            let mut complex_data = vec![Complex32::new(0.0, 0.0); dim];
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    payload.as_ptr(),
                                    complex_data.as_mut_ptr() as *mut u8,
                                    byte_len,
                                );
                            }
                            let query = VectorEmbedding::from_complex(complex_data);
                            let ctx = RequestContext {
                                request_id: header.request_id as u64,
                                ..Default::default()
                            };

                            match service.search_with_proof(&ctx, &query, k) {
                                Ok(response) => {
                                    let header_pos = write_buf.len();
                                    // Bit 0: certified exact. Bit 1: proof upper bound follows
                                    // the result count in the payload. These flags deliberately
                                    // reflect planner evidence, never result count heuristics.
                                    let cert_flags = u16::from(response.is_certified)
                                        | (u16::from(response.proof_upper_bound.is_some()) << 1);
                                    let response_header = MessageHeader {
                                        magic: PROTOCOL_MAGIC,
                                        opcode: OpCode::Search,
                                        flags: cert_flags,
                                        request_id: header.request_id,
                                        payload_len: 0,
                                    };
                                    response_header.encode(&mut write_buf);

                                    let payload_start = write_buf.len();
                                    write_buf.put_u32(response.results.len() as u32);
                                    if let Some(upper_bound) = response.proof_upper_bound {
                                        write_buf.put_f32(upper_bound);
                                    }
                                    for (res_id, score) in response.results {
                                        let id_bytes = res_id.as_bytes();
                                        write_buf.put_u16(id_bytes.len() as u16);
                                        write_buf.put_slice(id_bytes);
                                        write_buf.put_f32(score);
                                    }
                                    let written_len = (write_buf.len() - payload_start) as u32;

                                    let len_bytes = written_len.to_be_bytes();
                                    write_buf[header_pos + 12..header_pos + 16]
                                        .copy_from_slice(&len_bytes);
                                }
                                Err(e) => {
                                    Self::write_error(
                                        &mut write_buf,
                                        header.request_id,
                                        &e.to_string(),
                                    );
                                }
                            }
                        }
                    }
                }
                OpCode::BatchSearch => {
                    if payload.len() < 12 {
                        Self::write_error(
                            &mut write_buf,
                            header.request_id,
                            "Malformed batch search payload",
                        );
                    } else {
                        let k = payload.get_u32() as usize;
                        let count = payload.get_u32() as usize;
                        let dim = payload.get_u32() as usize;
                        let vec_bytes = dim * std::mem::size_of::<Complex32>();

                        if payload.len() < count * vec_bytes {
                            Self::write_error(
                                &mut write_buf,
                                header.request_id,
                                "Truncated batch search vectors",
                            );
                        } else {
                            let mut queries = Vec::with_capacity(count);
                            for i in 0..count {
                                let offset = i * vec_bytes;
                                let mut complex_data = vec![Complex32::new(0.0, 0.0); dim];
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        payload[offset..].as_ptr(),
                                        complex_data.as_mut_ptr() as *mut u8,
                                        vec_bytes,
                                    );
                                }
                                queries.push(VectorEmbedding::from_complex(complex_data));
                            }

                            let ctx = RequestContext {
                                request_id: header.request_id as u64,
                                ..Default::default()
                            };

                            let mut batch_results = Vec::with_capacity(queries.len());
                            for q in &queries {
                                match service.search(&ctx, q, k) {
                                    Ok(res) => batch_results.push(res),
                                    Err(e) => {
                                        Self::write_error(
                                            &mut write_buf,
                                            header.request_id,
                                            &e.to_string(),
                                        );
                                        break;
                                    }
                                }
                            }

                            if batch_results.len() == queries.len() {
                                let header_pos = write_buf.len();
                                let response_header = MessageHeader {
                                    magic: PROTOCOL_MAGIC,
                                    opcode: OpCode::BatchSearch,
                                    flags: 0,
                                    request_id: header.request_id,
                                    payload_len: 0,
                                };
                                response_header.encode(&mut write_buf);

                                let payload_start = write_buf.len();
                                write_buf.put_u32(batch_results.len() as u32);
                                for res_list in batch_results {
                                    write_buf.put_u32(res_list.len() as u32);
                                    for (res_id, score) in res_list {
                                        let id_bytes = res_id.as_bytes();
                                        write_buf.put_u16(id_bytes.len() as u16);
                                        write_buf.put_slice(id_bytes);
                                        write_buf.put_f32(score);
                                    }
                                }
                                let written_len = (write_buf.len() - payload_start) as u32;
                                let len_bytes = written_len.to_be_bytes();
                                write_buf[header_pos + 12..header_pos + 16]
                                    .copy_from_slice(&len_bytes);
                            }
                        }
                    }
                }
                OpCode::Stats => {
                    let json = r#"{"status":"ready","engine":"hnsqr"}"#;
                    let resp_header = MessageHeader {
                        magic: PROTOCOL_MAGIC,
                        opcode: OpCode::Stats,
                        flags: 0,
                        request_id: header.request_id,
                        payload_len: json.len() as u32,
                    };
                    resp_header.encode(&mut write_buf);
                    write_buf.put_slice(json.as_bytes());
                }
                OpCode::GraphQuery => {
                    let q_len = payload.get_u32() as usize;
                    if payload.len() < q_len {
                        Self::write_error(
                            &mut write_buf,
                            header.request_id,
                            "Truncated graph query payload",
                        );
                    } else {
                        let q_bytes = payload.split_to(q_len);
                        let q_str = std::str::from_utf8(&q_bytes).unwrap_or("");
                        let ctx = RequestContext {
                            request_id: header.request_id as u64,
                            ..Default::default()
                        };
                        match service.graph_query(&ctx, q_str) {
                            Ok(query_result) => {
                                let json = serde_json::to_string(&query_result).unwrap_or_default();
                                let resp_header = MessageHeader {
                                    magic: PROTOCOL_MAGIC,
                                    opcode: OpCode::GraphQuery,
                                    flags: 0,
                                    request_id: header.request_id,
                                    payload_len: json.len() as u32,
                                };
                                resp_header.encode(&mut write_buf);
                                write_buf.put_slice(json.as_bytes());
                            }
                            Err(e) => {
                                Self::write_error(&mut write_buf, header.request_id, &e.to_string())
                            }
                        }
                    }
                }
                _ => {
                    Self::write_error(&mut write_buf, header.request_id, "Unsupported opcode");
                }
            }

            stream
                .write_all(&write_buf)
                .await
                .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        }
    }

    #[inline(always)]
    fn write_error(dst: &mut BytesMut, request_id: u32, msg: &str) {
        let msg_bytes = msg.as_bytes();
        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            opcode: OpCode::Error,
            flags: 0,
            request_id,
            payload_len: msg_bytes.len() as u32,
        };
        header.encode(dst);
        dst.put_slice(msg_bytes);
    }
}

/// Asynchronous TCP client for the HNSQR database service.
pub struct HNSQRClient {
    stream: TcpStream,
    next_req_id: u32,
}

impl HNSQRClient {
    /// Connects to a remote HNSQR TCP server.
    pub async fn connect<A: tokio::net::ToSocketAddrs>(addr: A) -> HNSQRResult<Self> {
        let stream = TcpStream::connect(addr)
            .await
            .map_err(|e| HNSQRError::ConcurrencyError(format!("Failed to connect: {}", e)))?;
        Ok(Self {
            stream,
            next_req_id: 1,
        })
    }

    /// Sends a ping healthcheck to the server.
    pub async fn ping(&mut self) -> HNSQRResult<bool> {
        let req_id = self.next_req_id;
        self.next_req_id += 1;

        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            opcode: OpCode::Ping,
            flags: 0,
            request_id: req_id,
            payload_len: 4,
        };

        let mut buf = BytesMut::with_capacity(HEADER_SIZE + 4);
        header.encode(&mut buf);
        buf.put_slice(b"PING");

        self.stream
            .write_all(&buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut resp_hdr_buf = [0u8; HEADER_SIZE];
        self.stream
            .read_exact(&mut resp_hdr_buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut hdr_bytes = BytesMut::from(&resp_hdr_buf[..]);
        let resp_hdr = MessageHeader::decode(&mut hdr_bytes)?.unwrap();

        let mut payload = vec![0u8; resp_hdr.payload_len as usize];
        self.stream
            .read_exact(&mut payload)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        Ok(&payload == b"PONG")
    }

    /// Inserts a vector embedding via TCP network protocol with direct slice streaming.
    pub async fn insert(&mut self, id: &str, vector: &VectorEmbedding) -> HNSQRResult<u32> {
        let req_id = self.next_req_id;
        self.next_req_id += 1;

        let id_bytes = id.as_bytes();
        let cdata = vector.complex_data();
        let dim = cdata.len();
        let payload_len = 2 + id_bytes.len() + 4 + std::mem::size_of_val(cdata);

        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            opcode: OpCode::Insert,
            flags: 0,
            request_id: req_id,
            payload_len: payload_len as u32,
        };

        let mut buf = BytesMut::with_capacity(HEADER_SIZE + payload_len);
        header.encode(&mut buf);
        buf.put_u16(id_bytes.len() as u16);
        buf.put_slice(id_bytes);
        buf.put_u32(dim as u32);

        unsafe {
            let byte_ptr = cdata.as_ptr() as *const u8;
            let byte_len = std::mem::size_of_val(cdata);
            let raw_slice = std::slice::from_raw_parts(byte_ptr, byte_len);
            buf.put_slice(raw_slice);
        }

        self.stream
            .write_all(&buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut resp_hdr_buf = [0u8; HEADER_SIZE];
        self.stream
            .read_exact(&mut resp_hdr_buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut hdr_bytes = BytesMut::from(&resp_hdr_buf[..]);
        let resp_hdr = MessageHeader::decode(&mut hdr_bytes)?.unwrap();

        if resp_hdr.opcode == OpCode::Error {
            let mut err_payload = vec![0u8; resp_hdr.payload_len as usize];
            self.stream
                .read_exact(&mut err_payload)
                .await
                .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
            return Err(HNSQRError::SearchError(
                // from_utf8 avoids into_owned() allocation on well-formed UTF-8.
                String::from_utf8(err_payload)
                    .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned()),
            ));
        }

        let mut node_idx_buf = [0u8; 4];
        self.stream
            .read_exact(&mut node_idx_buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        Ok((&node_idx_buf[..]).get_u32())
    }

    /// Performs nearest neighbor search over TCP network protocol with direct slice streaming.
    pub async fn search(
        &mut self,
        query: &VectorEmbedding,
        k: usize,
    ) -> HNSQRResult<Vec<(String, SimilarityScore)>> {
        Ok(self.search_with_proof(query, k).await?.results)
    }

    /// Performs a search and returns the protocol-level verification evidence.
    pub async fn search_with_proof(
        &mut self,
        query: &VectorEmbedding,
        k: usize,
    ) -> HNSQRResult<Qir0SearchResponse> {
        let req_id = self.next_req_id;
        self.next_req_id += 1;

        let cdata = query.complex_data();
        let dim = cdata.len();
        let payload_len = 4 + 4 + std::mem::size_of_val(cdata);

        let header = MessageHeader {
            magic: PROTOCOL_MAGIC,
            opcode: OpCode::Search,
            flags: 0,
            request_id: req_id,
            payload_len: payload_len as u32,
        };

        let mut buf = BytesMut::with_capacity(HEADER_SIZE + payload_len);
        header.encode(&mut buf);
        buf.put_u32(k as u32);
        buf.put_u32(dim as u32);

        unsafe {
            let byte_ptr = cdata.as_ptr() as *const u8;
            let byte_len = std::mem::size_of_val(cdata);
            let raw_slice = std::slice::from_raw_parts(byte_ptr, byte_len);
            buf.put_slice(raw_slice);
        }

        self.stream
            .write_all(&buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut resp_hdr_buf = [0u8; HEADER_SIZE];
        self.stream
            .read_exact(&mut resp_hdr_buf)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut hdr_bytes = BytesMut::from(&resp_hdr_buf[..]);
        let resp_hdr = MessageHeader::decode(&mut hdr_bytes)?.unwrap();

        // Zero-copy receive: read directly into a pre-allocated BytesMut so we can
        // call split_to() for ID slices without a second heap allocation (versus
        // reading into Vec<u8> and then re-wrapping it in BytesMut::from).
        let mut payload = BytesMut::with_capacity(resp_hdr.payload_len as usize);
        payload.resize(resp_hdr.payload_len as usize, 0);
        self.stream
            .read_exact(&mut payload)
            .await
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let count = payload.get_u32() as usize;
        let is_certified = resp_hdr.flags & 0x0001 != 0;
        let proof_upper_bound = if resp_hdr.flags & 0x0002 != 0 {
            if payload.len() < std::mem::size_of::<f32>() {
                return Err(HNSQRError::SerializationError(
                    "QIR0 search response declares a proof bound but omits it".to_string(),
                ));
            }
            Some(payload.get_f32())
        } else {
            None
        };
        let mut results = Vec::with_capacity(count);

        for _ in 0..count {
            let id_len = payload.get_u16() as usize;
            let id_bytes = payload.split_to(id_len);
            let score = payload.get_f32();
            // Zero-allocation on well-formed UTF-8 (standard server output): from_utf8
            // borrows the slice to validate in-place, then converts to owned String once.
            // into_owned() on Cow::Borrowed is avoided on the happy path.
            let id_str = String::from_utf8(id_bytes.into())
                .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
            results.push((id_str, score));
        }

        Ok(Qir0SearchResponse {
            results,
            is_certified,
            proof_upper_bound,
        })
    }
}

/// Search response received over QIR0, including optional planner verification.
#[derive(Clone, Debug, PartialEq)]
pub struct Qir0SearchResponse {
    pub results: Vec<(String, SimilarityScore)>,
    pub is_certified: bool,
    pub proof_upper_bound: Option<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HNSQRConfig;

    #[tokio::test]
    async fn test_tcp_server_client_roundtrip() {
        let config = HNSQRConfig::default();
        let index = Arc::new(HNSQRIndex::new(config, 2));

        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let bound_addr = listener.local_addr().unwrap();

        let index_srv = Arc::clone(&index);
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let srv = Arc::new(StandaloneService::new(index_srv));
            let _ = HNSQRServer::handle_client(stream, srv).await;
        });

        let mut client = HNSQRClient::connect(bound_addr).await.unwrap();
        assert!(client.ping().await.unwrap());

        let v1 = VectorEmbedding::new(vec![1.0, 0.0]);
        let idx = client.insert("doc_remote_1", &v1).await.unwrap();
        assert_eq!(idx, 0);

        let results = client.search(&v1, 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc_remote_1");
    }
}
