//! # GERMEN
//!
//! Train a small language model **from scratch**, in pure Rust, on an ordinary CPU.
//!
//! The network is a GPT-style decoder (RMSNorm, rotary positions, squared-ReLU MLP) over a byte-level
//! vocabulary, written on [Burn](https://burn.dev) with its pure-Rust `flex` backend: no Python,
//! no CUDA, no C/C++ toolchain, nothing to install besides the binary itself.
//!
//! * [`config`] — architecture and size presets
//! * [`model`] — the network
//! * [`data`] — corpus, batches, built-in demo text
//! * [`train`] — the training loop
//! * [`generate`] — sampling text from a trained model
//! * [`checkpoint`] — saving and loading
//! * [`bench`] — measuring training speed on the current machine
#![warn(missing_docs)]

pub mod bench;
pub mod checkpoint;
pub mod config;
pub mod data;
pub mod generate;
pub mod model;
pub mod sys;
pub mod tokenizer;
pub mod train;

use burn::prelude::Device;

/// Version of the GERMEN library (and of the checkpoints it writes).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Device for inference and evaluation: the pure-Rust CPU backend, no gradient tracking.
pub fn cpu_device() -> Device {
    Device::flex()
}

/// Device for training: the pure-Rust CPU backend with automatic differentiation.
pub fn train_device() -> Device {
    Device::flex().autodiff()
}
