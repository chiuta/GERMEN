//! Gradient helpers: the global gradient norm and clipping by it.
//!
//! Burn's built-in clipping works on each parameter tensor separately, which changes the
//! direction of the update. The usual recipe for training language models is different: scale
//! *all* gradients by the same factor when their joint (global) L2 norm exceeds a threshold. This
//! keeps the direction of the update and only shortens it.

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;

struct SumOfSquares<'a> {
    grads: &'a GradientsParams,
    total: f64,
}

impl ModuleVisitor for SumOfSquares<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(g) = self.grads.get::<D>(param.id) {
            self.total += (g.clone() * g).sum().into_scalar::<f32>() as f64;
        }
    }
}

struct Scale<'a> {
    grads: &'a mut GradientsParams,
    factor: f32,
}

impl ModuleVisitor for Scale<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if let Some(g) = self.grads.remove::<D>(param.id) {
            self.grads
                .register::<D>(param.id, g.mul_scalar(self.factor));
        }
    }
}

/// L2 norm of all gradients in `grads` that belong to parameters of `model`.
pub fn global_norm<M: Module>(grads: &GradientsParams, model: &M) -> f32 {
    let mut visitor = SumOfSquares { grads, total: 0.0 };
    model.visit(&mut visitor);
    visitor.total.sqrt() as f32
}

/// Scales every gradient by `max_norm / norm` when the global `norm` exceeds `max_norm`.
///
/// Returns the norm *before* clipping (which may be `NaN` or infinite if the gradients are; the
/// gradients are then left untouched and the caller should stop).
pub fn clip_global_norm<M: Module>(grads: &mut GradientsParams, model: &M, max_norm: f32) -> f32 {
    let norm = global_norm(grads, model);
    if norm.is_finite() && norm > max_norm && norm > 0.0 {
        let mut visitor = Scale {
            grads,
            factor: max_norm / norm,
        };
        model.visit(&mut visitor);
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;
    use burn::optim::GradientsAccumulator;

    fn batch(
        device: &Device,
        rows: std::ops::Range<usize>,
        t: usize,
    ) -> (Tensor<2, Int>, Tensor<2, Int>) {
        let n = rows.len();
        let make = |shift: usize| {
            let data: Vec<i32> = rows
                .clone()
                .flat_map(|r| (0..t).map(move |c| ((r * 37 + c * 11 + shift) % 256) as i32))
                .collect();
            Tensor::<2, Int>::from_data(TensorData::new(data, [n, t]), device)
        };
        (make(0), make(1))
    }

    fn all_grads(grads: &GradientsParams, model: &crate::model::Gpt) -> Vec<f32> {
        struct Collect<'a>(&'a GradientsParams, Vec<f32>);
        impl ModuleVisitor for Collect<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
                if let Some(g) = self.0.get::<D>(param.id) {
                    self.1.extend(g.into_data().try_to_vec::<f32>().unwrap());
                }
            }
        }
        let mut c = Collect(grads, Vec::new());
        model.visit(&mut c);
        c.1
    }

    #[test]
    fn global_norm_matches_a_manual_computation() {
        let device = Device::flex().autodiff();
        let model = Preset::Nano.config(16).init(&device);
        let (x, y) = batch(&device, 0..3, 16);
        let grads = GradientsParams::from_grads(model.loss(x, y).backward(), &model);
        let manual = all_grads(&grads, &model)
            .iter()
            .map(|&g| g as f64 * g as f64)
            .sum::<f64>()
            .sqrt() as f32;
        let got = global_norm(&grads, &model);
        assert!(got > 0.0);
        assert!((got - manual).abs() <= 1e-3 * manual, "{got} vs {manual}");
    }

    #[test]
    fn clipping_scales_all_gradients_equally_and_only_when_needed() {
        let device = Device::flex().autodiff();
        let model = Preset::Nano.config(16).init(&device);
        let (x, y) = batch(&device, 0..3, 16);
        let original = GradientsParams::from_grads(model.loss(x, y).backward(), &model);
        let norm = global_norm(&original, &model);

        // Threshold above the norm: untouched.
        let (x, y) = batch(&device, 0..3, 16);
        let mut grads = GradientsParams::from_grads(model.loss(x, y).backward(), &model);
        let before = clip_global_norm(&mut grads, &model, norm * 2.0);
        assert!((before - norm).abs() < 1e-4 * norm);
        assert!((global_norm(&grads, &model) - norm).abs() < 1e-4 * norm);

        // Threshold at half the norm: the norm becomes the threshold and the direction is kept.
        let (x, y) = batch(&device, 0..3, 16);
        let mut grads = GradientsParams::from_grads(model.loss(x, y).backward(), &model);
        let limit = norm / 2.0;
        clip_global_norm(&mut grads, &model, limit);
        let after = global_norm(&grads, &model);
        assert!((after - limit).abs() < 1e-3 * limit, "{after} vs {limit}");
        let (a, b) = (all_grads(&original, &model), all_grads(&grads, &model));
        for (p, q) in a.iter().zip(&b).filter(|(p, _)| p.abs() > 1e-6) {
            assert!((q / p - 0.5).abs() < 1e-3, "ratio {}", q / p);
        }
    }

    #[test]
    fn accumulated_micro_batches_equal_one_big_batch() {
        let device = Device::flex().autodiff();
        let model = Preset::Nano.config(16).init(&device);

        let (x, y) = batch(&device, 0..8, 16);
        let whole = GradientsParams::from_grads(model.loss(x, y).backward(), &model);

        let mut acc = GradientsAccumulator::new();
        for rows in [0..4, 4..8] {
            let (x, y) = batch(&device, rows, 16);
            let loss = model.loss(x, y).div_scalar(2.0);
            acc.accumulate(&model, GradientsParams::from_grads(loss.backward(), &model));
        }
        let summed = acc.grads();

        let (a, b) = (all_grads(&whole, &model), all_grads(&summed, &model));
        assert_eq!(a.len(), b.len());
        let scale = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let worst = a
            .iter()
            .zip(&b)
            .fold(0.0f32, |m, (p, q)| m.max((p - q).abs()));
        assert!(
            worst < 1e-4 * scale.max(1e-3),
            "worst {worst}, scale {scale}"
        );
    }
}
