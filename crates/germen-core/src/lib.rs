//! # GERMEN
//!
//! Train a small language model **from scratch**, in pure Rust, on an ordinary CPU.
//!
//! The network is a GPT-style decoder (RMSNorm, rotary positions, squared-ReLU MLP) over bytes or a
//! trained byte-level BPE vocabulary, written on [Burn](https://burn.dev) with its pure-Rust `flex`
//! backend: no Python, no CUDA, no C/C++ toolchain, nothing to install besides the binary itself.
//!
//! * [`config`] — architecture and size presets
//! * [`model`] — the network, with a key/value cache for generation
//! * [`tokenizer`] and [`bpe`] — bytes or a trainable BPE vocabulary
//! * [`data`] — corpus, batches, built-in demo text
//! * [`train`] — the training loop (AdamW, accumulation, resume)
//! * [`gradients`] — global-norm clipping
//! * [`generate`] — sampling text from a trained model
//! * [`checkpoint`] — saving and loading
//! * [`mod@bench`] — measuring training speed (and its scaling with threads) on this machine
//! * [`sys`] — machine description and the worker pool a command should run in
#![warn(missing_docs)]
// The only `unsafe` in the workspace is one call to the Windows API in `sys::windows`, which allows it
// for itself; anywhere else it is a compile error.
#![deny(unsafe_code)]

pub mod bench;
pub mod bpe;
pub mod checkpoint;
pub mod config;
pub mod data;
mod fsutil;
pub mod generate;
pub mod gradients;
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
