/* holosphere/src/vector/mod.rs */
//!▫~•◦-------------------------------‣
//! # Vector Representation, Folding & Quantization Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod folding;
pub mod gpu_tensor;
pub mod hypercube;
pub mod inference;
pub mod polar;
pub mod quantization;
pub mod rotary;

pub use folding::{ComplexSliceCast, ComplexWeaver, GatewayRouter, create_http_router, run_http_server};
pub use gpu_tensor::{GpuDeviceConfig, GpuExecutionDevice, GpuPrecision, GpuTensorAccelerator};
pub use hypercube::{CoordinateND, HypercubeBoundingBox, HypercubeSnapshot, HypercubeTensorSpace};
pub use inference::{InProcessModelEmbedder, InferenceModelConfig, ModelArchitecture};
pub use polar::CircularAngularMetric;
pub use quantization::PolarQuantizedVector;
pub use rotary::RotaryPhaseTransformer;
